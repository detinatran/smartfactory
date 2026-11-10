"""Engine-disjoint splits, fit-only normalization and strictly past-ending windows."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SETTINGS = [f"setting_{i}" for i in range(1, 4)]
SENSORS = [f"sensor_{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
COLUMNS = ["unit", "cycle", *SETTINGS, *[f"sensor_{i}" for i in range(1, 22)]]
EXPECTED = {
    "FD001": (100, 100, 20631, 13096),
    "FD002": (260, 259, 53759, 33991),
    "FD003": (100, 100, 24720, 16596),
    # NASA's bundled readme swaps the FD004 train/test counts; these are file-derived.
    "FD004": (249, 248, 61249, 41214),
}


def read_frame(path: Path) -> pd.DataFrame:
    values = np.loadtxt(path)
    if values.ndim != 2 or values.shape[1] != 26 or not np.isfinite(values).all():
        raise ValueError(f"Expected finite [N,26] data: {path}")
    if not np.array_equal(values[:, :2], values[:, :2].astype(int)):
        raise ValueError("Unit IDs and cycles must be integers")
    df = pd.DataFrame(values, columns=COLUMNS)
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    for _, group in df.groupby("unit", sort=True):
        if not np.array_equal(group.cycle, np.arange(1, len(group) + 1)):
            raise ValueError(f"Cycles must be ordered, consecutive and begin at 1: {path}")
    return df


def validate_data(raw: Path) -> dict:
    manifest_path = raw.parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, record in manifest["files"].items():
        if hashlib.sha256((raw / name).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
    summary = {}
    for subset, (n_train, n_test, train_rows, test_rows) in EXPECTED.items():
        train = read_frame(raw / f"train_{subset}.txt")
        test = read_frame(raw / f"test_{subset}.txt")
        rul = np.loadtxt(raw / f"RUL_{subset}.txt", ndmin=1)
        actual = (train.unit.nunique(), test.unit.nunique(), len(train), len(test))
        if actual != (n_train, n_test, train_rows, test_rows):
            raise ValueError(f"Unexpected dataset dimensions for {subset}: {actual}")
        if len(rul) != n_test or not np.isfinite(rul).all() or np.any(rul < 0):
            raise ValueError(f"Invalid endpoint RUL labels: {subset}")
        if not np.array_equal(np.sort(test.unit.unique()), np.arange(1, n_test + 1)):
            raise ValueError("Test label ordering requires consecutive unit IDs")
        summary[subset] = dict(
            train_engines=n_train, test_engines=n_test, train_rows=train_rows, test_rows=test_rows
        )
    return summary


def split_engines(units, seed: int) -> dict[str, list[int]]:
    """50/10/15/10/15 percent; the official test is outside these five partitions."""
    ids = np.sort(np.unique(units)).copy()
    if len(ids) < 20:
        raise ValueError("At least 20 engines are required for five independent partitions")
    np.random.default_rng(seed).shuffle(ids)
    boundaries = (np.array([0.50, 0.60, 0.75, 0.85]) * len(ids)).astype(int)
    parts = np.split(ids, boundaries)
    names = ["fit", "validation", "selector", "calibration", "development"]
    return {name: sorted(part.tolist()) for name, part in zip(names, parts, strict=True)}


class Normalizer:
    """Condition-wise sensor scaling, using only backbone-fitting engines."""

    def __init__(self, n_regimes=1, seed=42):
        self.n_regimes = n_regimes
        self.seed = seed
        self.settings_scaler = StandardScaler()
        self.clusterer = None

    def fit(self, df):
        settings = self.settings_scaler.fit_transform(df[SETTINGS])
        if self.n_regimes > 1:
            self.clusterer = KMeans(n_clusters=self.n_regimes, n_init=10, random_state=self.seed)
            regimes = self.clusterer.fit_predict(settings)
        else:
            regimes = np.zeros(len(df), dtype=int)
        sensors = df[SENSORS].to_numpy()
        self.means = np.stack([sensors[regimes == i].mean(0) for i in range(self.n_regimes)])
        std = np.stack([sensors[regimes == i].std(0) for i in range(self.n_regimes)])
        self.scales = np.where(std > 1e-6, std, 1.0)
        return self

    def regimes(self, df):
        settings = self.settings_scaler.transform(df[SETTINGS])
        return (
            self.clusterer.predict(settings)
            if self.clusterer is not None
            else np.zeros(len(df), int)
        )

    def transform(self, df):
        regimes = self.regimes(df)
        sensor = df[SENSORS].to_numpy()
        missing = np.isnan(sensor)
        values = np.nan_to_num((sensor - self.means[regimes]) / self.scales[regimes], nan=0.0)
        settings = self.settings_scaler.transform(df[SETTINGS])
        return np.concatenate([values, settings, missing.astype(float)], axis=1).astype(np.float32)


def corrupt(df, normalizer, scenario, seed):
    """Corrupt full trajectories BEFORE windowing, with onset independent of failure time.

    The same cycle has the same corruption in every overlapping window. The oracle
    corruption flag is never exposed to a selector; only actual missingness is observed.
    """
    if scenario == "clean":
        return df.copy()
    from .faults import apply_fault, parse_fault

    fault = parse_fault(scenario)
    if fault:  # unseen-fault library; v1 scenarios below are unchanged
        kind, magnitude, n_channels = fault
        out = df.copy()
        for unit, group in out.groupby("unit", sort=True):
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(unit)]))
            channels = rng.choice(len(SENSORS), size=n_channels, replace=False)
            onset = int(rng.integers(40, 81))
            extra = np.random.default_rng(np.random.SeedSequence([seed, int(unit), 77]))
            cycles = group.cycle.to_numpy()
            hit = (cycles >= onset) & (cycles < onset + 60)
            regimes = normalizer.regimes(group)
            for j in channels:
                out.loc[group.index, SENSORS[j]] = apply_fault(
                    group[SENSORS[j]].to_numpy(), hit, kind, normalizer.scales[regimes, j],
                    normalizer.means[regimes, j], extra, magnitude)
        return out
    if scenario not in ("dropout", "drift", "dropout_strong", "drift_strong"):
        raise ValueError(f"Unknown scenario: {scenario}")
    out = df.copy()
    strong = scenario.endswith("_strong")
    for unit, group in out.groupby("unit", sort=True):
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(unit)]))
        channels = rng.choice(len(SENSORS), size=3 if strong else 1, replace=False)
        onset = int(rng.integers(40, 81))
        cycles = group.cycle.to_numpy()
        if scenario.startswith("dropout"):
            affected = (cycles >= onset) & (cycles < onset + (80 if strong else 40))
            out.loc[group.index[affected], [SENSORS[j] for j in channels]] = np.nan
        else:
            regimes = normalizer.regimes(group)
            ramp = np.clip((cycles - onset) / 100.0, 0.0, 1.0)
            for j in channels:
                sign = rng.choice([-1, 1])
                shift = sign * (4.0 if strong else 2.0) * ramp * normalizer.scales[regimes, j]
                out.loc[group.index, SENSORS[j]] = group[SENSORS[j]].to_numpy() + shift
    return out


@dataclass
class Windows:
    x: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame


def make_windows(df, normalizer, window=30, stride=5, cap=125.0, endpoint_rul=None):
    """For official test use one last window per unit; left-pad short histories.

    Training/development windows begin when a full window is available. Targets
    are clipped for regression; raw labels are retained for asymmetric-risk evaluation.
    """
    xs, ys, rows = [], [], []
    for unit, group in df.groupby("unit", sort=True):
        group = group.sort_values("cycle")
        values = normalizer.transform(group)
        cycles = group.cycle.to_numpy()
        if endpoint_rul is None:
            endpoints = list(range(window - 1, len(group), stride))
            if not endpoints:
                raise ValueError(f"Training engine {unit} is shorter than the window")
            if endpoints[-1] != len(group) - 1:
                endpoints.append(len(group) - 1)
            final_rul = 0.0
        else:
            endpoints = [len(group) - 1]
            final_rul = float(endpoint_rul[int(unit)])
        for end in endpoints:
            x = values[max(0, end - window + 1) : end + 1]
            if len(x) < window:
                x = np.pad(x, ((window - len(x), 0), (0, 0)), mode="edge")
            raw_rul = float(cycles[-1] - cycles[end] + final_rul)
            xs.append(x)
            ys.append(min(raw_rul, cap))
            rows.append({"unit": int(unit), "cycle": int(cycles[end]), "rul_raw": raw_rul})
    if not xs:
        raise ValueError("No windows produced")
    return Windows(np.stack(xs), np.array(ys, dtype=np.float32), pd.DataFrame(rows))


def engine_weights(units):
    """Each engine contributes equal total weight, regardless of trajectory length."""
    _, inverse, counts = np.unique(units, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse]
    return w / w.mean()
