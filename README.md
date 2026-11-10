# When Sensors Fail, Uncertainty Looks Away

Code, data records and statistics for the paper *When Sensors Fail, Uncertainty Looks Away: Fault-Induced Bias and One-Sided Selective RUL Prediction*.

Sensor faults often shift all ensemble members in the same direction. The ensemble spread barely reacts to such a common shift, so spread-based abstention keeps the resulting optimistic remaining-useful-life (RUL) errors. We instead estimate the probability of an optimistic error, P(ŷ − y > δ | x), with a gradient-boosting classifier and abstain when it is high.

The repository contains:
- the full experiment pipeline;
- the exact split IDs and per-run configurations;
- the raw results behind every table;
- scripts that regenerate all tables, figures and statistical tests of the paper.

The regenerated tables are byte-identical to the paper's.

## Repository layout

```
.
├── src/selective_rul/     Python package: data loading, splits, fault injection, backbones, selectors, baselines, metrics
├── scripts/               experiment pipeline: data preparation → training → baselines → fault evaluation → CPU replay
├── configs/               training configurations (C-MAPSS, external datasets, smoke test)
├── tests/                 unit tests (pytest)
│
├── paper/                 regenerate the paper from results/ (no GPU, no raw data needed)
│   ├── run_all.py         runs everything below in order (~1–2 min)
│   ├── fig1_mechanism.py, figS1_units.py
│   ├── main_tests.py      main test family, Holm over 100 tests
│   ├── hier_bootstrap.py  seed × unit bootstrap
│   ├── tables.py          Tables 1–4
│   ├── exploratory_tests.py
│   ├── host/              scripts that exported run_records/ and splits/unit_map.csv from the training runs
│   └── outputs/           generated tables (.tex), figures and statistics
│
├── results/               raw per-seed / per-fold / per-unit results (inputs of paper/)
├── splits/                exact unit IDs of every split, unit-name map, consistency checks
├── run_records/           config and checkpoint hashes of every run
├── data/fig1/             per-window predictions of the run shown in Fig. 1
└── docs/                  hyperparameters, evaluation protocol and statistics
```

## Quick start: reproduce the paper's tables and tests

The environment is pinned in `uv.lock` (Python 3.11–3.13).

```bash
uv sync
uv run python paper/run_all.py     # writes paper/outputs/{tables,figures,stats}
uv run pytest                      # 63 unit tests
```

| Paper item | Script | Output |
|---|---|---|
| Fig. 1 | `fig1_mechanism.py` | `outputs/figures/mechanism_example.pdf` |
| Table 1: mechanism | `tables.py` | `outputs/tables/mech.tex` |
| Table 2: what the score uses, fault detectors | `tables.py` | `outputs/tables/uses.tex` |
| Table 3: frozen thresholds | `tables.py` | `outputs/tables/frozen.tex`, `outputs/stats/table3_coverage_shift.csv` |
| Table 4: paired differences | `main_tests.py`, `hier_bootstrap.py`, `tables.py` | `outputs/tables/paired_hier.tex` |
| Main test family (100 tests) | `main_tests.py` | `outputs/stats/main_family.csv`, `main_family_differences.csv` |
| Exploratory tests quoted in the text | `exploratory_tests.py` | `outputs/stats/exploratory.csv`, `exploratory_differences.csv`, `selectivity.csv` |
| Fig. S1: all 39 units of the Fig. 1 run | `figS1_units.py` | `outputs/figures/figS1_unit_distribution.pdf`, `outputs/stats/figS1_unit_mechanism.csv` |

Every test row records:
- the 20 paired differences;
- the Wilcoxon statistic W;
- the counts of zero and tied differences;
- exact and normal-approximation p-values;
- the Holm-adjusted p-value (main family).

## Data records

| Folder / file | Content |
|---|---|
| `splits/splits_long.csv` | one row per (dataset, fold, seed, split, unit), with original unit names; 120 split groups |
| `splits/split_checks.json` | automatic checks (see below) |
| `splits/unit_map.csv` | unit code → original name, source fold, regime, number of records, life |
| `run_records/run_configs.csv` | full configuration and checkpoint SHA-256 prefixes of the 240 backbone runs |
| `run_records/run_manifest.json.gz` | config, splits, completion time and hashes of all 760 runs (backbones, fault evaluation, baselines) |
| `results/matched_per_seed.csv.gz` | risk at matched coverage 0.5–1.0, per dataset × backbone × seed × scenario × method |
| `results/frozen_per_seed.csv.gz` | risk and achieved coverage with thresholds frozen on clean calibration units |
| `results/replay_scores_per_fold.csv.gz` | CPU replay per run and fold: all Table 2 scores, risk, coverage, decision flips, rejection of fault-induced errors |
| `results/unit_masses.csv.gz` | accepted and optimistic mass per unit (input of the hierarchical bootstrap) |
| `results/mechanism_summary.csv` | prediction shift vs. spread on identical clean/faulty windows (Table 1) |

`split_checks.json` verifies, over all 240 backbone runs, that:
- splits are disjoint;
- GRU and TCN use the same split for each (dataset, fold, seed);
- all 240 fault-evaluation runs use their backbone's split;
- N-CMAPSS is evaluated on exactly the 39 official test units;
- batteries are evaluated on batch 3;
- each bearing is evaluated exactly once per seed.

## Experiment pipeline (full re-run, GPU recommended)

| Step | Script | Notes |
|---|---|---|
| 1. Data | `download_data.py`, `prepare_external.py --root data/external`, `prepare_battery.py --root data/external/battery` | public datasets: C-MAPSS, N-CMAPSS, FEMTO/PRONOSTIA, XJTU-SY, Severson et al. (2019) |
| 2. Backbones | e.g. `run_matrix.py --config configs/ext_capped.json --subsets FD001 FD002 FD003 FD004 --models gru tcn --seeds 42 … 51`, `run_external.py --config configs/external.json --data-dir data/external/processed --models gru tcn --seeds 42 … 51 --out runs/external` | 3-member ensembles; trains the one-sided selector on selector units |
| 3. Baselines | `extra_baselines.py --runs runs/…` | SelectiveNet, MC dropout, Gaussian-NLL ensemble, CQR |
| 4. Unseen faults | `fault_study.py`, `fault_mechanism.py`, `unit_masses.py` | six faults, fault augmentation, clean/faulty mechanism, per-unit masses |
| 5. CPU replay | `ablation_v5.py --runs runs/… --device cpu` | Table 2 scores, including both fault detectors |
| 6. Summaries | `summarize_extension.py`, `summarize_external.py` | per-seed tables in `results/` |
| Simulation | `sim_common_shift.py` | common vs. member-specific shift |

Fault injection is described in `src/selective_rul/faults.py` and in [docs/hyperparameters.md](docs/hyperparameters.md):
- the fault is applied per unit after splitting, on raw signals, before windowing;
- each fault hits 3 channels;
- the random choices are seeded by `SeedSequence([seed + 10000, unit])`.

## Documentation

- [docs/hyperparameters.md](docs/hyperparameters.md): data, splits, backbones, selectors, baselines, faults.
- [docs/protocol.md](docs/protocol.md): what was pre-specified, metric and thresholds, aggregation, CPU replay, statistics.
