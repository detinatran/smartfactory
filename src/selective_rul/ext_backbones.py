"""Backbones for the EXT2 pilot, re-implemented from the papers (no code copied from the original repositories).

LipschitzRNNRUL   Erichson et al., "Lipschitz Recurrent Neural Networks", ICLR 2021 (arXiv:2006.12070):
                  h_{t+1} = h_t + eps * (A h_t + tanh(W h_t + U x_t + b)),
                  A = (1 - beta) (M_A + M_A^T) + beta (M_A - M_A^T) - gamma I, and W built the same way from M_W
                  (forward-Euler discretisation of the continuous-time unit).
FiLMEnsembleGRU   Turkoglu et al., "FiLM-Ensemble: Probabilistic Deep Learning via Feature-wise Linear Modulation",
                  NeurIPS 2022 (arXiv:2206.00050), adapted to a 2-layer GRU: one shared network; after each recurrent
                  layer a parameter-free LayerNorm followed by member-specific FiLM (gamma_m * h + beta_m). Diversity
                  comes from random initialisation of the FiLM parameters (gamma ~ N(1, s^2), beta ~ N(0, s^2)).
                  `member_views` exposes the M members as separate modules, so the existing ensemble code
                  (predict_details, MC dropout, selector features) works unchanged.
"""

import torch
from torch import nn

from .backbones import PointReadout


def _structured(m, beta, gamma):
    eye = torch.eye(m.shape[0], device=m.device, dtype=m.dtype)
    return (1 - beta) * (m + m.T) + beta * (m - m.T) - gamma * eye


class LipschitzRNNRUL(PointReadout):
    def __init__(self, input_dim, hidden, dropout, cap, beta=0.75, gamma=0.001, eps=0.1):
        super().__init__()
        self.cap, self.beta, self.gamma, self.eps = cap, beta, gamma, eps
        self.m_a = nn.Parameter(torch.randn(hidden, hidden) / hidden ** 0.5)
        self.m_w = nn.Parameter(torch.randn(hidden, hidden) / hidden ** 0.5)
        self.inp = nn.Linear(input_dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def features(self, x):
        a = _structured(self.m_a, self.beta, self.gamma)
        w = _structured(self.m_w, self.beta, self.gamma)
        u = self.inp(x)  # [B, T, H]
        h = x.new_zeros(x.shape[0], a.shape[0])
        for t in range(x.shape[1]):
            h = h + self.eps * (h @ a.T + torch.tanh(h @ w.T + u[:, t]))
        return self.drop(h)


class FiLMEnsembleGRU(nn.Module):
    def __init__(self, input_dim, hidden, dropout, cap, members=8, init_std=0.5):
        super().__init__()
        self.cap, self.members = cap, members
        self.gru1 = nn.GRU(input_dim, hidden, batch_first=True)
        self.gru2 = nn.GRU(hidden, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.gamma = nn.Parameter(1 + init_std * torch.randn(2, members, hidden))
        self.beta = nn.Parameter(init_std * torch.randn(2, members, hidden))
        self.head = nn.Linear(hidden, 1)
        self.readout = "sigmoid"

    def _film(self, h, layer, member):
        if member is None:  # h: [M, B, T, H]
            g, b = self.gamma[layer], self.beta[layer]  # [M, H]
            return self.norm(h) * g[:, None, None] + b[:, None, None]
        return self.norm(h) * self.gamma[layer, member] + self.beta[layer, member]

    def features(self, x, member=None):
        """member=None: all members, returns [M, B, H]; otherwise [B, H] for one member."""
        h, _ = self.gru1(x)
        if member is None:
            h = self._film(h.unsqueeze(0).expand(self.members, *h.shape), 0, None)
            m, bsz, t, d = h.shape
            h, _ = self.gru2(self.drop(h).reshape(m * bsz, t, d))
            h = self._film(h.reshape(m, bsz, t, d), 1, None)
            return h[:, :, -1]
        h = self._film(h, 0, member)
        h, _ = self.gru2(self.drop(h))
        return self._film(h, 1, member)[:, -1]

    def forward_all(self, x):
        """[B, M] member predictions."""
        return (torch.sigmoid(self.head(self.features(x)).squeeze(-1)) * self.cap).T

    def member_views(self):
        return [FiLMMember(self, m) for m in range(self.members)]


class FiLMMember(PointReadout):
    """One member of a FiLMEnsembleGRU (shares all weights with the parent)."""

    def __init__(self, parent, member):
        super().__init__()
        self.parent, self.member = parent, member
        self.cap = parent.cap

    @property
    def head(self):
        return self.parent.head

    def features(self, x):
        return self.parent.features(x, self.member)
