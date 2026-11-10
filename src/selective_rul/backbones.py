"""Additional RUL backbones, implemented locally without changing the legacy TCN.

PatchTransformer is a channel-mixing patch encoder, not a PatchTST reproduction.
MambaReference is a sequential PyTorch implementation of the Mamba-1 selective
SSM recurrence; it does not use the official fused kernels or pretrained weights.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PointReadout(nn.Module):
    # "sigmoid": bounded in (0, cap), the v1 default. "softplus": unbounded above;
    # cap is then only an output scale (used by the uncapped-target ablation).
    readout = "sigmoid"

    def regress(self, features):
        logits = self.head(features).squeeze(-1)
        if self.readout == "softplus":
            return F.softplus(logits) * self.cap
        return torch.sigmoid(logits) * self.cap

    def forward(self, x):
        return self.regress(self.features(x))

    def forward_outputs(self, x):
        return {"point": self(x), "quantiles": None}


class RecurrentRUL(PointReadout):
    def __init__(self, input_dim, hidden, dropout, cap, layers=2, kind="gru"):
        super().__init__()
        self.cap = cap
        recurrent = nn.GRU if kind == "gru" else nn.LSTM
        self.encoder = recurrent(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def features(self, x):
        sequence, _ = self.encoder(x)
        return sequence[:, -1]


class PatchTransformerRUL(PointReadout):
    def __init__(
        self,
        input_dim,
        hidden,
        dropout,
        cap,
        window,
        layers=2,
        heads=4,
        patch_length=10,
        patch_stride=5,
    ):
        super().__init__()
        self.cap, self.window = cap, window
        self.patch_length, self.patch_stride = patch_length, patch_stride
        if window < patch_length or (window - patch_length) % patch_stride:
            raise ValueError("Patch windows must tile exactly and include the current cycle")
        self.n_patches = (window - patch_length) // patch_stride + 1
        if self.n_patches < 2:
            raise ValueError("At least two patches are required for temporal attention")
        self.projection = nn.Linear(input_dim * patch_length, hidden)
        self.position = nn.Parameter(torch.empty(1, self.n_patches, hidden))
        nn.init.normal_(self.position, std=0.02)
        # Construct layers separately: no identical initialization from encoder cloning.
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    hidden,
                    heads,
                    dim_feedforward=4 * hidden,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def features(self, x):
        if x.shape[1] != self.window:
            raise ValueError(f"Expected {self.window} observed cycles, received {x.shape[1]}")
        # [B, patches, channels, patch_length] -> one token per observed patch.
        patches = x.unfold(1, self.patch_length, self.patch_stride).flatten(2)
        sequence = self.projection(patches) + self.position
        for block in self.blocks:
            sequence = block(sequence)
        # All tokens lie within the observed history; no t+1 data is supplied.
        return self.norm(sequence.mean(dim=1))


def selective_scan_reference(u, delta, a, b, c, skip):
    """Mamba-1 recurrence: state_t = exp(delta_t*A)*state + delta_t*B_t*u_t.

    Shapes: u/delta [batch,time,inner], A [inner,state], B/C [batch,time,state].
    This is intentionally a sequential reference implementation, not a fast scan.
    """
    state = u.new_zeros(u.shape[0], u.shape[2], a.shape[1])
    outputs = []
    for t in range(u.shape[1]):
        step = delta[:, t, :, None]
        state = torch.exp(step * a) * state + step * b[:, t, None, :] * u[:, t, :, None]
        outputs.append((state * c[:, t, None, :]).sum(-1) + skip * u[:, t])
    return torch.stack(outputs, dim=1)


class MambaReferenceBlock(nn.Module):
    def __init__(self, hidden, state_size=8, expand=2, conv_width=4, dropout=0.1):
        super().__init__()
        inner = hidden * expand
        rank = max(1, math.ceil(hidden / 16))
        self.rank, self.state_size, self.padding = rank, state_size, conv_width - 1
        self.norm = nn.LayerNorm(hidden)
        self.in_projection = nn.Linear(hidden, 2 * inner, bias=False)
        self.conv = nn.Conv1d(inner, inner, conv_width, groups=inner, padding=self.padding)
        self.select = nn.Linear(inner, rank + 2 * state_size, bias=False)
        self.delta_projection = nn.Linear(rank, inner)
        # Positive, small initial step sizes; stable negative state transition spectrum.
        nn.init.uniform_(self.delta_projection.weight, -(rank**-0.5), rank**-0.5)
        dt = torch.exp(torch.empty(inner).uniform_(math.log(0.001), math.log(0.1)))
        with torch.no_grad():
            self.delta_projection.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.a_log = nn.Parameter(torch.arange(1, state_size + 1).float().log().repeat(inner, 1))
        self.skip = nn.Parameter(torch.ones(inner))
        self.out_projection = nn.Linear(inner, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        u, gate = self.in_projection(self.norm(x)).chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2))[:, :, : x.shape[1]].transpose(1, 2))
        delta_low, b, c = torch.split(
            self.select(u), [self.rank, self.state_size, self.state_size], dim=-1
        )
        delta = F.softplus(self.delta_projection(delta_low))
        scanned = selective_scan_reference(u, delta, -torch.exp(self.a_log), b, c, self.skip)
        return x + self.dropout(self.out_projection(scanned * F.silu(gate)))


class MambaReferenceRUL(PointReadout):
    def __init__(self, input_dim, hidden, dropout, cap, layers=2, state_size=8):
        super().__init__()
        self.cap = cap
        self.projection = nn.Linear(input_dim, hidden)
        self.blocks = nn.Sequential(
            *[
                MambaReferenceBlock(hidden, state_size=state_size, dropout=dropout)
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def features(self, x):
        return self.norm(self.blocks(self.projection(x))[:, -1])


class QuantileRUL(nn.Module):
    """Add ordered RAW-RUL quantiles while retaining the SAME capped point task.

    q10/q50/q90 are not confidence bounds. The independent point head stays capped
    for matched baseline comparisons; auxiliary quantiles never use clipped labels.
    """

    levels = (0.1, 0.5, 0.9)

    def __init__(self, backbone, hidden, cap):
        super().__init__()
        self.backbone, self.cap = backbone, cap
        self.quantile_head = nn.Linear(hidden, 3)
        nn.init.normal_(self.quantile_head.weight, std=0.01)
        with torch.no_grad():
            self.quantile_head.bias.copy_(torch.tensor([-2.0, -0.5, -0.5]))

    def forward_outputs(self, x):
        h = self.backbone.features(x)
        increments = F.softplus(self.quantile_head(h)) * self.cap
        return {"point": self.backbone.regress(h), "quantiles": increments.cumsum(dim=-1)}

    def forward(self, x):
        return self.forward_outputs(x)["point"]


def pinball_loss(quantiles, raw_targets):
    levels = quantiles.new_tensor(QuantileRUL.levels)
    residual = raw_targets[:, None] - quantiles
    return torch.maximum(levels * residual, (levels - 1) * residual).mean(dim=-1)
