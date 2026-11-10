"""Small causal temporal CNN ensemble; no future observations or unit IDs as inputs."""

import copy
import math
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .backbones import (
    MambaReferenceRUL,
    PatchTransformerRUL,
    PointReadout,
    QuantileRUL,
    RecurrentRUL,
    pinball_loss,
)
from .data import engine_weights

MODEL_NAMES = ("tcn", "gru", "lstm", "patch_transformer", "mamba_ref", "lipschitz_rnn")
HEAD_NAMES = ("point", "quantile")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, 3, padding=self.padding, dilation=dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv(x)[:, :, : -self.padding]
        return self.activation(x + self.dropout(self.activation(h)))


class RULModel(PointReadout):
    """Legacy TCN: unchanged parameter keys and default 30-cycle behavior."""

    def __init__(self, input_dim, hidden=32, dropout=0.1, cap=125.0, window=30):
        super().__init__()
        self.cap = cap
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden, 1),
            *[
                CausalBlock(hidden, 2**i, dropout)
                for i in range(max(4, math.ceil(math.log2((window + 1) / 2))))
            ],
        )
        self.head = nn.Linear(hidden, 1)

    def features(self, x):
        return self.encoder(x.transpose(1, 2))[:, :, -1]


def build_model(input_dim, cfg):
    name = cfg.get("model", "tcn")
    head = cfg.get("head", "point")
    common = dict(
        input_dim=input_dim, hidden=cfg["hidden"], dropout=cfg["dropout"], cap=cfg["rul_cap"]
    )
    if name == "tcn":
        model = RULModel(**common, window=cfg["window"])
    elif name in ("gru", "lstm"):
        model = RecurrentRUL(**common, kind=name, layers=cfg.get("layers", 2))
    elif name == "patch_transformer":
        model = PatchTransformerRUL(
            **common,
            window=cfg["window"],
            layers=cfg.get("layers", 2),
            heads=cfg.get("attention_heads", 4),
            patch_length=cfg.get("patch_length", 10),
            patch_stride=cfg.get("patch_stride", 5),
        )
    elif name == "lipschitz_rnn":
        from .ext_backbones import LipschitzRNNRUL

        model = LipschitzRNNRUL(**common, beta=cfg.get("lip_beta", 0.75), gamma=cfg.get("lip_gamma", 0.001),
                                eps=cfg.get("lip_eps", 0.1))
    elif name == "mamba_ref":
        model = MambaReferenceRUL(
            **common, layers=cfg.get("layers", 2), state_size=cfg.get("state_size", 8)
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")
    model.readout = cfg.get("point_readout", "sigmoid")
    if head == "quantile":
        return QuantileRUL(model, cfg["hidden"], cfg["rul_cap"])
    if head != "point":
        raise ValueError(f"Unknown head: {head}")
    return model


@torch.inference_mode()
def predict_details(models, x, device, batch_size=512):
    points, quantiles = [], []
    for model in models:
        model.eval()
        member_points, member_quantiles = [], []
        for start in range(0, len(x), batch_size):
            output = model.forward_outputs(
                torch.from_numpy(x[start : start + batch_size]).to(device)
            )
            member_points.append(output["point"].cpu().numpy())
            if output["quantiles"] is not None:
                member_quantiles.append(output["quantiles"].cpu().numpy())
        points.append(np.concatenate(member_points))
        if member_quantiles:
            quantiles.append(np.concatenate(member_quantiles))
    # Mean of member quantile estimates, NOT quantiles of an ensemble mixture.
    return np.stack(points, axis=1), np.stack(quantiles).mean(0) if quantiles else None


@torch.inference_mode()
def predict(models, x, device, batch_size=512):
    outputs = []
    for model in models:
        model.eval()
        pieces = []
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            pieces.append(model(batch).cpu().numpy())
        outputs.append(np.concatenate(pieces))
    return np.stack(outputs, axis=1)


def train_member(train, validation, cfg, device, seed):
    seed_everything(seed)
    model = build_model(train.x.shape[-1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    weights = engine_weights(train.meta.unit.to_numpy()).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(train.x),
        torch.from_numpy(train.y),
        torch.from_numpy(weights),
        torch.tensor(train.meta.rul_raw.to_numpy(), dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    val_weights = engine_weights(validation.meta.unit.to_numpy())
    # Uncapped ablation: regress raw RUL (train and early stopping) instead of capped RUL.
    raw_target = cfg.get("point_target", "capped") == "raw"
    val_target = validation.meta.rul_raw.to_numpy() if raw_target else validation.y
    history, best, stale = [], float("inf"), 0
    best_state = None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses, point_losses = [], []
        for x, y, w, raw_y in loader:
            x, y, w, raw_y = x.to(device), y.to(device), w.to(device), raw_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_outputs(x)
            target = raw_y if raw_target else y
            point_loss = (w * ((output["point"] - target) / cfg["rul_cap"]).square()).mean()
            loss = point_loss
            if output["quantiles"] is not None:
                auxiliary = (w * pinball_loss(output["quantiles"], raw_y) / cfg["rul_cap"]).mean()
                loss = loss + cfg.get("quantile_weight", 0.2) * auxiliary
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            point_losses.append(float(point_loss.detach().cpu()))
        ensemble, quantiles = predict_details([model], validation.x, device)
        pred = ensemble[:, 0]
        rmse = float(np.sqrt(np.average((pred - val_target) ** 2, weights=val_weights)))
        if not np.isfinite(rmse):
            raise RuntimeError("Non-finite validation loss")
        record = dict(
            epoch=epoch,
            train_scaled_mse=float(np.mean(point_losses)),
            train_loss=float(np.mean(losses)),
            val_engine_rmse=rmse,
        )
        if quantiles is not None:
            val_pinball = pinball_loss(
                torch.from_numpy(quantiles), torch.tensor(validation.meta.rul_raw.to_numpy())
            )
            record["val_raw_pinball"] = float(np.average(val_pinball.numpy(), weights=val_weights))
        history.append(record)
        print(f"member_seed={seed} epoch={epoch:02d} val_engine_rmse={rmse:.3f}", flush=True)
        if rmse < best:
            best, stale = rmse, 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    return model, history
