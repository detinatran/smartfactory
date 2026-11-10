"""Unseen sensor-fault library (protocol: FAULT_PROTOCOL.md). Never used to train v1 selectors.

Every fault hits 3 channels over one interval [onset, onset + duration). Per channel
(scale = fit-set spread of that channel, mean = regime/unit reference level):
  stuck        : value frozen at the last pre-interval reading
  bias         : +/- 3 scale added
  noise        : N(0, (2 scale)^2) added
  intermittent : each record missing (NaN) with probability 0.5
  spike        : each record, with probability 0.1, gets +/- 6 scale added
  gain         : mean + 1.5 (value - mean)
"""

import numpy as np

FAULTS = ("stuck", "bias", "noise", "intermittent", "spike", "gain")
N_CHANNELS = 3
DEFAULT_MAGNITUDE = {"bias": 3.0, "noise": 2.0, "spike": 6.0, "gain": 1.5}


def parse_fault(scenario):
    """'bias' -> ('bias', None, 3); 'bias@6' -> magnitude 6; 'stuck#6' -> 6 channels. None if not a fault."""
    name, magnitude, channels = scenario, None, N_CHANNELS
    if "#" in name:
        name, n = name.split("#", 1)
        channels = int(n)
    if "@" in name:
        name, m = name.split("@", 1)
        magnitude = float(m)
    return (name, magnitude, channels) if name in FAULTS else None


def apply_fault(values, hit, kind, scale, mean, rng, magnitude=None):
    """values: 1-D float array of one channel of one unit (time-ordered); hit: bool mask."""
    out = values.astype(float).copy()
    m = DEFAULT_MAGNITUDE.get(kind) if magnitude is None else magnitude
    if not hit.any():
        return out
    idx = np.flatnonzero(hit)
    if kind == "stuck":
        first = idx[0]
        out[idx] = out[first - 1] if first > 0 else out[first]
    elif kind == "bias":
        out[idx] += rng.choice([-1, 1]) * m * scale[idx]
    elif kind == "noise":
        out[idx] += rng.normal(0.0, m, len(idx)) * scale[idx]
    elif kind == "intermittent":
        out[idx[rng.random(len(idx)) < 0.5]] = np.nan
    elif kind == "spike":
        chosen = idx[rng.random(len(idx)) < 0.1]
        out[chosen] += rng.choice([-1, 1], len(chosen)) * m * scale[chosen]
    elif kind == "gain":
        out[idx] = mean[idx] + m * (out[idx] - mean[idx])
    else:
        raise ValueError(kind)
    return out
