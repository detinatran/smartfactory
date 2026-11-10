# Hyperparameters

All values were fixed before the results and never tuned on evaluation data. Per-run values are in [`run_records/run_configs.csv`](../run_records/run_configs.csv): one row per backbone run, 240 rows, with the SHA-256 prefix of each checkpoint.

Within a dataset, every config field is identical across runs except `seed`, `model` and `fold`. Source code is in [`src/selective_rul/`](../src/selective_rul/).

## Data, windows and targets

| | C-MAPSS FD001–FD004 | N-CMAPSS DS01–DS08c | Bearings (FEMTO + XJTU-SY) | Battery (Severson 2019) |
|---|---|---|---|---|
| Time step | cycle | flight | record | cycle |
| Window / stride | 30 / 5 | 10 / 1 | 30 / 5 | 30 / 10 |
| Target | RUL capped at 125 cycles | remaining-life fraction | remaining-life fraction | remaining-life fraction |
| Optimistic threshold δ | 20 cycles | 0.16 (= 20/125) | 0.16 | 0.16 |
| Normalisation (fit units only) | per operating regime (KMeans, 6 regimes on FD002/FD004, 1 otherwise) | per flight class (z-score) | minus the unit's mean over its first 10 records, divided by the fit-set spread | as bearings |
| Inputs | 14 sensors + 3 settings + missingness flags | 19 | 34 (h/v rms, peak, p2p in log scale, …) | 7 |

## Splits (exact unit IDs: [`splits/splits_long.csv`](../splits/splits_long.csv))

Splits are made per physical unit and are disjoint. GRU and TCN use the same split for a given (dataset, fold, seed); [`splits/split_checks.json`](../splits/split_checks.json) verifies this.

| | Pool | fit / validation / selector / calibration | Evaluation |
|---|---|---|---|
| C-MAPSS | official training engines of each subset | 50 / 10 / 15 / 10 % | 15 % held-out "development" engines (the official test set is not used) |
| N-CMAPSS | 60 official development units | rest / 15 / 20 / 15 % | the 39 official test units (identical for all seeds) |
| Bearings | the other operating conditions | rest / 15 / 20 / 15 % | the held-out condition (6 folds; each bearing evaluated once per seed) |
| Battery | batches 1–2 | rest / 10 / 20 / 30 % | batch 3 (40 cells) |

How the permutations are drawn:
- C-MAPSS: `numpy.random.default_rng(seed).shuffle` (`data.split_engines`).
- External datasets: `default_rng(SeedSequence([seed, 7919]))` (`external.split_units`).

Seeds are 42–51.

## Backbones (point ensembles)

| | Value |
|---|---|
| GRU | 2 layers, hidden 64, dropout 0.1, last hidden state → linear → sigmoid readout scaled to the cap |
| TCN | 1×1 input conv to 64 channels; causal residual blocks with kernel 3, dilations 1, 2, 4, 8, GELU, dropout 0.1; last time step → linear → sigmoid readout |
| Ensemble | 3 members; member *m* uses seed + 100·*m* |
| Loss | unit-weighted squared error on the target divided by the cap (each unit has equal total weight) |
| Optimiser | AdamW, learning rate 1e-3, weight decay 1e-4, batch 128, gradient clipping 1.0 |
| Schedule | at most 40 epochs; early stopping with patience 8 on unit-weighted validation RMSE, keeping the best epoch |
| Hardware | trained on one GPU; Table 2 scores were re-computed on CPU from the same checkpoints |

## Selectors

| Score | Definition |
|---|---|
| **One-sided (ours)** | `HistGradientBoostingClassifier` for 1[ŷ − y > δ]. 100 iterations, max_leaf_nodes 7, min_samples_leaf 25, l2_regularization 2.0, no early stopping, random_state = seed. `engine_weights` sample weights, no class balancing. Trained on selector units × {clean, dropout, drift}. |
| Features | ensemble mean, ensemble spread, elapsed time (C-MAPSS: cycle/125; others: log(1 + step), the observed step index, never total life); per channel: window mean, std, last − first, missing rate |
| Symmetric event classifier | as ours, target 1[\|ŷ − y\| > δ] |
| Fault detector (window statistics / all features) | as ours, target 1[window differs from its clean version], features = window statistics only, or all features |
| Fault augmentation (`asym_aug`) | as ours, also trained on the 5 unseen faults other than the evaluated one (leave-one-fault-out); corruption groups have equal total weight |
| Ensemble spread | std of the 3 member predictions |
| MC dropout | dropout active at test time, T = 20 passes per member; mean over members of the MC std |
| Deep ensemble (NLL) | 3 Gaussian-NLL members (mean + log-variance); total predictive std |
| CQR lower bound | 3 pinball-loss members at q = 0.1; score = shared prediction − mean q10 |
| SelectiveNet | selection head on the backbone, target coverage 0.9, λ = 32, α = 0.5, squared loss; score = 1 − g(x) |

All selectors rank the **same** shared ensemble predictions; the baselines provide only a score.

Thresholds are chosen in one of two ways:
- **matched:** set on the evaluation scores at 90% unit-balanced coverage;
- **frozen:** set on clean calibration units at 90% and then kept fixed.

Ties are broken by a fixed per-(seed, unit, cycle) random key (`selection.tie_breakers`).

## Faults ([`faults.py`](../src/selective_rul/faults.py), `data.corrupt`, `external.corrupt_external`)

Faults are injected per unit after splitting, on the raw signal, before windowing. Every window that contains a given time step therefore sees the same corruption.

The random choices are seeded as follows:
- `SeedSequence([run_seed + 10000, unit])` draws the 3 channels and the onset;
- `SeedSequence([run_seed + 10000, unit, 77])` draws the fault-specific randomness.

| | C-MAPSS | Other datasets |
|---|---|---|
| Interval | onset ~ U{40, …, 80} cycles, length 60 cycles; truncated at end of life | onset ~ U[0.2, 0.4] of life, length 0.3 of life; always inside the trajectory |
| Scale σ | per-regime std on fit units | std on fit units |
| Reference level (gain) | per-regime mean | the unit's own mean before onset |

Life fractions are used only to time the injected fault. They are never a model input.

| Fault (never used for training, except in `asym_aug`) | Definition on the 3 affected channels |
|---|---|
| stuck | hold the last reading before onset |
| bias | add ±3σ (random sign per channel) |
| noise | add N(0, (2σ)²) |
| intermittent | each record missing (NaN) with probability 0.5 |
| spike | each record, with probability 0.1, gets ±6σ added |
| gain | mean + 1.5·(x − mean) |

The seen corruptions are used to train the selectors and the fault detector:

| Corruption | C-MAPSS | Other datasets |
|---|---|---|
| dropout | 1 channel missing for 40 cycles (strong: 3 channels, 80 cycles) | ≈ n/14 channels missing for 0.2 of life (strong: 3n/14, 0.4) |
| drift | 1 channel, ramp to ±2σ over 100 cycles (strong: 3 channels, ±4σ) | ≈ n/14 channels, ramp to ±2σ over 0.5 of life (strong: ±4σ) |
