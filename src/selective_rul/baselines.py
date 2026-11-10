"""Additional selective-prediction baselines for RUL regression (protocol: BASELINES_PROTOCOL.md).

All of them only produce a SCORE (higher = abstain first); selective metrics are always computed
on the shared point-ensemble predictions so that every selector ranks the same predictions.

  selectivenet  : SelectiveNet (Geifman & El-Yaniv 2019) for regression: selection head g, target
                  coverage 0.9, lambda=32, alpha=0.5, squared loss; score = 1 - g(x)
  mc_dropout    : dropout kept active at test time in the already-trained ensemble members,
                  T=20 passes; score = mean over members of the MC standard deviation
  deep_ens_nll  : 3 Gaussian-NLL members (mean + log-variance); score = total predictive std
                  sqrt(mean sigma^2 + var of means)  (deep ensembles, Lakshminarayanan et al. 2017)
  cqr_lower     : 3 lower-quantile (q=0.1, pinball) members; score = shared prediction - mean q10
                  (a constant conformal correction would not change this ranking)
Deep Gamblers is classification-only and has no direct RUL analogue; deep_ens_nll replaces it.
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import engine_weights
from .model import build_model, seed_everything


class HeadNet(nn.Module):
    def __init__(self, backbone, extra):
        super().__init__()
        self.backbone = backbone
        self.extra = nn.Linear(backbone.head.in_features, extra) if extra else None

    def forward(self, x):
        h = self.backbone.features(x)
        point = self.backbone.regress(h)
        return point, (self.extra(h) if self.extra is not None else None)


def _loss(kind, point, extra, y, w, cap):
    err = (point - y) / cap
    if kind == "selectivenet":
        g = torch.sigmoid(extra[:, 0])
        aux = torch.sigmoid(extra[:, 1])  # auxiliary prediction in (0, 1) x cap
        sel_risk = (g * w * err.square()).sum() / (g * w).sum().clamp_min(1e-6)
        coverage = (g * w).sum() / w.sum()
        penalty = 32.0 * torch.clamp(0.9 - coverage, min=0).square()
        aux_loss = (w * (aux - y / cap).square()).mean()
        return 0.5 * (sel_risk + penalty) + 0.5 * aux_loss
    if kind == "deep_ens_nll":
        log_var = extra[:, 0].clamp(-12, 6)
        return (w * 0.5 * (log_var + err.square() * torch.exp(-log_var))).mean()
    if kind == "cqr_lower":
        residual = (y - point) / cap
        return (w * torch.maximum(0.1 * residual, (0.1 - 1) * residual)).mean()
    raise ValueError(kind)


def train_head_model(kind, train, validation, cfg, device, seed):
    seed_everything(seed)
    extra = {"selectivenet": 2, "deep_ens_nll": 1, "cqr_lower": 0}[kind]
    model = HeadNet(build_model(train.x.shape[-1], dict(cfg, head="point")), extra).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    cap = float(cfg["rul_cap"])
    w_train = engine_weights(train.meta.unit.to_numpy()).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train.x), torch.from_numpy(train.y),
                      torch.from_numpy(w_train)),
        batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed))
    xv = torch.from_numpy(validation.x).to(device)
    yv = torch.from_numpy(validation.y).to(device)
    wv = torch.from_numpy(engine_weights(validation.meta.unit.to_numpy()).astype(np.float32)).to(device)
    best, stale, best_state = float("inf"), 0, None
    for _ in range(cfg["epochs"]):
        model.train()
        for x, y, w in loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            optimizer.zero_grad(set_to_none=True)
            point, ext = model(x)
            loss = _loss(kind, point, ext, y, w, cap)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            parts = [model(xv[i:i + 1024]) for i in range(0, len(xv), 1024)]
            point = torch.cat([p[0] for p in parts])
            ext = torch.cat([p[1] for p in parts]) if parts[0][1] is not None else None
            value = float(_loss(kind, point, ext, yv, wv, cap))
        if not np.isfinite(value):
            raise RuntimeError(f"non-finite validation loss for {kind}")
        if value < best:
            best, stale = value, 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def head_outputs(model, x, device, batch=1024):
    points, extras = [], []
    for i in range(0, len(x), batch):
        p, e = model(torch.from_numpy(x[i:i + batch]).to(device))
        points.append(p.cpu().numpy())
        if e is not None:
            extras.append(e.cpu().numpy())
    return np.concatenate(points), (np.concatenate(extras) if extras else None)


@torch.no_grad()
def mc_dropout_std(models, x, device, passes=20, batch=1024, seed=0):
    torch.manual_seed(seed)
    stds = []
    for model in models:
        model.train()  # activates dropout; no batch-norm layers in these backbones
        draws = []
        for _ in range(passes):
            draws.append(np.concatenate([
                model(torch.from_numpy(x[i:i + batch]).to(device)).cpu().numpy()
                for i in range(0, len(x), batch)]))
        model.eval()
        stds.append(np.std(np.stack(draws), axis=0))
    return np.mean(stds, axis=0)


def baseline_scores(fitted, ensemble_models, x, shared_prediction, cap, device, seed):
    """fitted: dict kind -> list of trained HeadNet models."""
    scores = {}
    point, ext = head_outputs(fitted["selectivenet"][0], x, device)
    scores["selectivenet"] = 1.0 - 1.0 / (1.0 + np.exp(-ext[:, 0]))
    means, variances = [], []
    for model in fitted["deep_ens_nll"]:
        point, ext = head_outputs(model, x, device)
        means.append(point / cap)
        variances.append(np.exp(np.clip(ext[:, 0], -12, 6)))
    means, variances = np.stack(means), np.stack(variances)
    scores["deep_ens_nll"] = np.sqrt(variances.mean(0) + means.var(0))
    q10 = np.mean([head_outputs(m, x, device)[0] for m in fitted["cqr_lower"]], axis=0)
    scores["cqr_lower"] = shared_prediction - q10
    scores["mc_dropout"] = mc_dropout_std(ensemble_models, x, device, seed=seed)
    return scores
