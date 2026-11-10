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
  - the hierarchical bootstrap.

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
