# Evaluation protocol and statistics

## What was pre-specified

The following were fixed in dated written protocols **before any results existed**:
- datasets, splits and windows;
- backbones and hyperparameters;
- the method: a one-sided classifier on all features;
- the baselines;
- the fault library and its seeds;
- the frozen-threshold protocol at 90% coverage as the **primary** protocol, with matched coverage as a diagnostic.

The fault study (six unseen faults, fault augmentation) was also fixed before its results.

The following were added later.
- **After the first results:** the paired Wilcoxon tests with Holm correction on risk at 90% matched coverage. The paper reports them as its main family of 100 tests.
- **Later still** (exploratory, unadjusted):
  - the fault detectors;
  - the lightweight score variants;
  - the unit-level mechanism analysis (Fig. S1);
  - the hierarchical bootstrap;
  - the uncertainty decomposition (MC dropout, learned bias and variance, Gaussian scores).

No backbone, selector or hyperparameter was changed after results were seen. The official C-MAPSS test set was used once, for the single frozen-threshold run quoted in the paper; it is not part of any table.

## Metric and thresholds

- **Risk:** the unit-balanced optimistic-error rate among accepted windows, where an optimistic error is ŷ − y > δ. Each unit has equal total weight, whatever its trajectory length.
- **Matched coverage:** the threshold is set on the evaluation scores to accept 90% of the unit-balanced mass. This compares rankings.
- **Frozen threshold:** the threshold is set on clean calibration units at 90% and kept fixed under faults. This is the deployment view, and achieved coverage is always reported next to risk.
- **Aggregation:**
  - C-MAPSS subsets are macro-averaged.
  - The 32 bearings (6 held-out operating conditions) are pooled as one unit-balanced population. Per-fold files are combined with unit-count weights ([`paper/common.py`](../paper/common.py), `per_seed`).
  - N-CMAPSS and batteries each have one evaluation set.

## CPU replay (Table 2 and exploratory tests)

[`scripts/ablation_v5.py`](../scripts/ablation_v5.py) reloads the **same checkpoints and splits**. It then:
- recomputes the ensemble predictions on CPU;
- refits every Table 2 score on selector units with the fixed hyperparameters;
- uses evaluation labels only to compute metrics.

CPU and GPU predictions differ only in floating point: the spread's risk matches the main pipeline within 0.06 points per configuration.

The refitted one-sided classifier is therefore close to, but not bit-identical with, the original: single-window scores differ by up to 0.40 (column `v1_reproduction_max_abs_diff`). Aggregate risk agrees within 0.05 points on every dataset:

| Dataset | CPU replay | Main pipeline |
|---|---|---|
| C-MAPSS | 9.37 | 9.39 |
| N-CMAPSS | 12.49 | 12.50 |
| Bearings | 23.29 | 23.29 |
| Battery | 4.47 | 4.42 |

The fault detector differs from the one-sided classifier **only in its target**: 1[window differs from its clean version] instead of 1[ŷ − y > δ]. Everything else is the same:
- model and fixed hyperparameters;
- training windows (selector units × clean, dropout, drift);
- features and unit weights;
- no class balancing.

## Statistics

**Unit of analysis.** Each pair is a (backbone, seed) configuration: GRU/TCN × 10 seeds = 20 pairs per dataset. Each value is the mean over the six unseen faults, or over the five standard scenarios (clean, dropout, drift, strong dropout, strong drift).

**Test.** Two-sided Wilcoxon signed-rank test, using `scipy.stats.wilcoxon` with its defaults:
- `zero_method='wilcox'`;
- exact null distribution when n ≤ 50 and there are no zeros.

**Ties.** None of the 100 main-family tests has a zero or a tied absolute difference, and neither does any exploratory test on risk or coverage, so all those p-values are exact. With 20/20 pairs of one sign, p = 2·2⁻²⁰ = 1.9·10⁻⁶.

Three exploratory comparisons of the rejection rate on bearings contain zero differences and therefore use the normal approximation; the `method` column flags them. None of the three is quoted in the paper. Over all 212 tests, no absolute differences are tied.

**Main family.** Holm correction covers 100 tests:
- 6 comparators × 4 datasets × 2 settings (standard shifts, unseen faults) × 2 metrics (risk at 90% matched coverage; AURC over coverage 0.5–1.0) = 96 tests;
- plus fault augmentation vs. ours on 4 datasets = 4 tests.

**Intervals.**
- Seed bootstrap: both backbones of a seed are resampled together; 10 000 replicates for the main family and 5 000 for exploratory tests.
- Hierarchical bootstrap (Table 4): seeds and then physical units are resampled, with 2 000 replicates. C-MAPSS uses one unit draw per (subset, seed) development set; the other datasets use one shared draw.

The configurations share evaluation units, so the hierarchical intervals are the more conservative ones.

## Fig. S1: is Fig. 1 typical?

Fig. 1 shows an N-CMAPSS test unit chosen after inspection: it ranks 3rd of 39 by mean upward prediction shift. Fig. S1 repeats the measurement for all 39 units of the same run (GRU, seed 42, bias fault). It shows that:
- the one-sided score rises in all 16 units whose prediction is pushed upward (median +0.53);
- it barely moves in the 23 units pushed downward (median −0.04);
- the spread rises in both groups (median +0.055 and +0.068).

## Uncertainty decomposition (exploratory)

[`scripts/uncertainty_v1.py`](../scripts/uncertainty_v1.py) replays the same checkpoints, splits, windows and faults on CPU (all 240 GRU/TCN runs). Every score is fitted on selector units × {clean, dropout, drift} with the fixed gradient-boosting settings, and evaluated on clean data and the six unseen faults.

| Score | Definition |
|---|---|
| `ensemble_std` | between-model epistemic uncertainty: std of the 3 member predictions (the ensemble spread) |
| `mc_dropout` | within-model epistemic uncertainty: MC dropout, T = 10 passes per member with dropout active; mean over members of the std |
| `mc_ensemble` | total epistemic uncertainty: std over all 3 × 10 stochastic predictions |
| `resid_sigma` | learned error scale √v(x), with v(x) = E[(e − b(x))² \| x] (Poisson-loss regressor on out-of-fold residuals, 2 folds by unit) |
| `gauss_unbiased` | 1 − Φ(δ / s), s² = v(x) + `mc_ensemble`²: Gaussian predictive distribution, variance only |
| `bias_only` | learned conditional bias b(x) = E[ŷ − y \| x] |
| `gauss_bias_aware` | 1 − Φ((δ − b(x)) / s): Gaussian predictive distribution with bias (Prop. 2) |
| `full` | the one-sided classifier of the paper |
| `full_plus_unc` | the same classifier with `mc_dropout` and `mc_ensemble` as extra features |

[`paper/uncertainty_tests.py`](../paper/uncertainty_tests.py) aggregates as in the main analysis and compares 20 (backbone, seed) pairs per dataset with unadjusted Wilcoxon tests and a seed bootstrap (5 000 replicates).

Risk (%) at 90% matched coverage, mean over the six unseen faults:

| Score | C-MAPSS | N-CMAPSS | Bearings | Battery |
|---|---|---|---|---|
| Ensemble spread | 11.16 | 19.51 | 24.08 | 5.52 |
| MC dropout | 11.50 | 19.84 | 24.28 | 6.14 |
| MC-dropout ensemble | 11.12 | 19.49 | 24.05 | 5.54 |
| Learned error scale | 12.33 | 16.74 | 25.29 | 5.50 |
| Gaussian, variance only | 12.33 | 17.71 | 25.16 | 5.45 |
| Learned bias | 10.60 | 12.16 | 23.57 | 3.48 |
| Gaussian, bias-aware | 10.24 | 12.18 | 23.47 | 3.56 |
| One-sided classifier | 9.37 | 12.49 | 23.29 | 4.47 |
| Classifier + MC-dropout features | 9.42 | 12.50 | 23.34 | 4.49 |

Findings:
- Other epistemic estimates do not improve on the spread. MC dropout is 0.2–0.6 points worse, and the MC-dropout ensemble is within 0.04 points.
- Adding MC-dropout features to the classifier changes risk by less than 0.1 points.
- On C-MAPSS, all three epistemic scores decrease as the fault raises the prediction: Spearman ρ between shift and score change is −0.31 to −0.44.
- Adding the learned bias b(x) to the Gaussian lowers risk by 1.7–5.5 points (18–20 of 20 pairs).
- The bias-aware Gaussian is close to the classifier. Its differences are −0.31 on N-CMAPSS, −0.91 on batteries, +0.87 on C-MAPSS and +0.18 (not significant) on bearings.
