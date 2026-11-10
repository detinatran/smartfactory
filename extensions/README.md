# Extensions: post-hoc uncertainty, FiLM-Ensemble and LipschitzRNN (exploratory)

These experiments go beyond the paper. They are **not** used in the paper's tables, and the paper's results are unchanged.
The goal was to test:
- whether post-hoc, statistically grounded uncertainty methods detect fault-induced optimistic RUL errors;
- whether blocks from two external architectures help:
  - FiLM-Ensemble (Turkoglu et al., 2022; github.com/prs-eth/FILM-Ensemble);
  - LipschitzRNN (Erichson et al., 2021; github.com/erichson/LipschitzRNN).

Notes on the external work:
- Both backbones were **re-implemented from the papers' equations** in `src/selective_rul/ext_backbones.py`. No code was
  copied from the original repositories; LipschitzRNN is GPL-3.0.
- SALAD (github.com/seokhyeonhong/salad) is a text-to-motion diffusion model. It has no uncertainty or RUL component, so it was
  not used.

Every round had a protocol written and frozen **before** it was run (`protocols/`). All analyses are exploratory. Rounds 3 and
4 were designed after earlier results had been seen; their protocols say so. Protocols and per-round findings are written in
Vietnamese.

## Rounds

| Round | Question | Runs | Protocol | Results |
|---|---|---|---|---|
| EXT2 pilot | 3 backbones (GRU, FiLM-GRU M=8, LipschitzRNN) × 8 uncertainty scores. C-MAPSS and N-CMAPSS, seeds 42–46 | 25 runs, 4 200 rows | `EXT2_PILOT_PROTOCOL.md` | `results/ext2_v1/` |
| EXT2 confirmation | Does "FiLM + bias-aware score beats GRU + same score" replicate on new seeds 47–51? | 25 runs, 2 800 rows | `EXT2_CONFIRM_PROTOCOL.md` (A) | `results/ext2_confirm/` |
| EXT2 extension | Same comparison on bearings (FEMTO, XJTU-SY) and batteries, seeds 42–46 | 35 runs, 5 880 rows | `EXT2_CONFIRM_PROTOCOL.md` (B) | `results/ext2_ext/` |
| EXT3 | Conformal variants and a combined score, on all 240 paper runs. Variants: weighted conformal under covariate shift, Mondrian, evidence-tuned Laplace, rank average | 240 runs, 13 440 rows | `EXT3_PROTOCOL.md` | `results/ext3_v1/` |
| EXT4 | Per-run choice between the paper's classifier and the bias-aware Gaussian, using calibration data only | 240 runs, 9 600 rows | `EXT4_PROTOCOL.md` | `results/ext4_v1/` |

No run failed in any round. Every round ran on CPU, replaying the paper's trained backbones; FiLM-GRU and LipschitzRNN were
trained on each run's own fit and validation units.

## Scores compared

All scores rank the same predictions.

| Score | Type | Definition |
|---|---|---|
| `ensemble_std`, `mc_dropout`, `mc_ensemble` | variance | ensemble spread; MC dropout (T = 10); MC dropout over all members |
| `laplace`, `laplace_ev` | variance, post hoc | last-layer GGN Laplace; `laplace_ev` picks the prior precision by marginal likelihood |
| `gauss_unbiased` | Gaussian | 1 − Φ(δ / s), with s² = v(x) + mc_ensemble² |
| `gauss_bias_aware` | Gaussian with learned bias | 1 − Φ((δ − b(x)) / s); b(x) and v(x) are HGB regressors fitted on out-of-fold residuals |
| `conformal` | split conformal | one-sided bound e ≤ b(x) + q·s(x), where q is the unit-weighted 90% quantile on clean calibration units |
| `conformal_w` | weighted conformal | as `conformal`, with likelihood-ratio weights from a calibration-vs-evaluation domain classifier |
| `mondrian` | conformal | q computed separately per quartile of the predicted RUL |
| `rank_avg` | combination | average of the rank of `full` and the rank of `gauss_bias_aware` |
| `select_cal` | selection | per run, whichever of `full` and `gauss_bias_aware` has the lower mean risk at coverage 0.5–0.9 on calibration units |
| `full` | the paper's selector | gradient-boosting classifier for ŷ − y > δ |

## Metric

The metric is the optimistic-error rate among accepted windows (ŷ − y > δ):
- δ = 20 cycles for C-MAPSS and N-CMAPSS, and 0.16 of life for bearings and batteries;
- every unit has equal total weight;
- it is measured at 90% matched coverage and averaged over the six unseen faults;
- aggregation is the same as in the paper.

## Main results

Optimistic-error rate (%) at 90% matched coverage, mean over six unseen faults, all 240 paper runs (EXT3, EXT4):

| Score | C-MAPSS | N-CMAPSS | Bearings | Batteries |
|---|---|---|---|---|
| ensemble_std | 11.16 | 19.51 | 24.08 | 5.52 |
| laplace_ev | 11.86 | 19.93 | 23.99 | 5.92 |
| conformal_w | 11.44 | 17.01 | 25.18 | 6.45 |
| conformal | 10.71 | 12.98 | 23.96 | 3.85 |
| gauss_bias_aware | 10.24 | 12.19 | 23.49 | **3.56** |
| rank_avg | 9.60 | **12.17** | **23.28** | 4.01 |
| select_cal | 9.54 | 12.20 | 23.35 | 3.82 |
| **full (paper)** | **9.37** | 12.49 | 23.29 | 4.47 |

Paired differences against the paper's selector (20 backbone–seed pairs, Wilcoxon, Holm within each round's main family):

| Comparison | C-MAPSS | N-CMAPSS | Bearings | Batteries |
|---|---|---|---|---|
| rank_avg − full (EXT3) | +0.22, p_H = 0.0005 | −0.32, p_H < 1e-4 | −0.01, n.s. | −0.45, p_H < 1e-4 |
| select_cal − full (EXT4) | +0.17, p_H = 0.0075 | −0.29, p_H = 0.030 | +0.07, n.s. | −0.64, p_H = 0.023 |

## Conclusions

1. **Variance-type uncertainty fails under sensor faults.** This holds for:
   - explicit ensembles, the implicit FiLM-Ensemble, MC dropout and post-hoc Laplace (including the evidence-tuned prior);
   - every backbone tested (GRU, TCN, FiLM-GRU, LipschitzRNN);
   - all four data families.

   MC dropout and Laplace differ from the ensemble spread by at most about 1 point, in no consistent direction. A learned
   residual variance without bias (`gauss_unbiased`) helps only on N-CMAPSS.
2. **The missing term is the conditional bias.** Against the ensemble spread, bias-aware scores lower the risk by:
   - 0.9–1.8 points on C-MAPSS;
   - 7.0–7.3 points on N-CMAPSS;
   - 1.1–2.0 points on batteries;
   - 0.6–0.8 points on bearings.
3. **No variant beats the paper's selector on all four families.** The pattern is the same in every round, and the
   differences between bias-aware scores are below 1 point:
   - the classifier is best on C-MAPSS;
   - the bias-aware Gaussian (or a combination with it) is best on N-CMAPSS and batteries;
   - bearings are tied.

   The paper's method is therefore unchanged.
4. **FiLM-Ensemble.**
   - Its implicit members shift together more than an explicit ensemble's. On C-MAPSS and N-CMAPSS, over 10 seeds, the ratio
     of prediction shift to spread is 1.4–1.8 times that of the GRU ensemble. This is the paper's common-shift mechanism.
   - "FiLM + bias-aware score is best" was a pilot finding. It replicated on N-CMAPSS (10/10 seeds) but **not** on C-MAPSS,
     and it reversed on batteries.
5. **LipschitzRNN** gave no consistent benefit. Its ε, β and γ were fixed in advance and not tuned.
6. **Conformal bounds.**
   - The bias-aware one-sided bound keeps about 90% coverage under faults on N-CMAPSS and batteries.
   - It drops to 82.5% on fault-affected C-MAPSS windows, and to 81–84% on bearings even without faults, because operating
     conditions differ between calibration and evaluation.
   - The EXT2 findings call the coverage "held under faults"; EXT3 corrects this (see `results/ext3_v1/FINDINGS.md`).
   - Weighted conformal fails when calibration and evaluation units do not overlap. Its weights degenerate, and q = ∞ on
     bearings and batteries.
7. **Frozen thresholds.** Bias-aware scores lower the risk at similar coverage on batteries (about 1.5–1.9% vs 2.64%) and on
   N-CMAPSS (about 7.8–7.9% vs 8.92%).

## Re-running

The scripts expect the training runs of the main pipeline (see the top-level README) and the processed external data. Each
command runs one shard; merge the per-run CSVs of `--out` into one `.csv.gz` before summarising.

```bash
# EXT2 pilot, confirmation and extension (GRU runs; FiLM-GRU and LipschitzRNN are trained here)
python scripts/ext2_pilot.py --runs runs/<cmapss_gru> runs/<external> --out runs/ext2_v1 --seeds 42 43 44 45 46 --device cpu
python scripts/ext2_pilot.py --runs ... --out runs/ext2_confirm --datasets cmapss ncmapss --backbones gru film_gru --seeds 47 48 49 50 51
python scripts/ext2_pilot.py --runs runs/<external> runs/<battery> --out runs/ext2_ext --datasets bearings battery --seeds 42 43 44 45 46
# EXT3 and EXT4 (all 240 GRU/TCN runs)
python scripts/ext3_conformal.py --runs runs/<all> --out runs/ext3_v1 --device cpu --passes 10
python scripts/ext4_select.py   --runs runs/<all> --out runs/ext4_v1 --device cpu --passes 10
```

Summaries and tests are regenerated from the stored results without a GPU or raw data, and are byte-identical to the stored
files:

```bash
python scripts/ext2_summary.py extensions/results/ext2_v1/all_ext2_v1.csv.gz        # same for ext2_confirm, ext2_ext
python scripts/ext2_confirm_tests.py extensions/results/ext2_confirm/all_ext2_confirm.csv.gz extensions/results/ext2_v1/all_ext2_v1.csv.gz
python scripts/ext3_summary.py extensions/results/ext3_v1/all_ext3_v1.csv.gz
python scripts/ext4_summary.py extensions/results/ext4_v1/all_ext4_v1.csv.gz
```

## Files

| Path | Content |
|---|---|
| `src/selective_rul/ext_backbones.py` | `LipschitzRNNRUL`; `FiLMEnsembleGRU` (shared 2-layer GRU, LayerNorm, per-member FiLM after each layer) |
| `scripts/ext2_pilot.py`, `ext3_conformal.py`, `ext4_select.py` | experiment scripts, one CSV per run |
| `scripts/ext2_summary.py`, `ext2_confirm_tests.py`, `ext3_summary.py`, `ext4_summary.py` | aggregation and paired tests |
| `results/<round>/all_*.csv.gz` | raw per-run, per-scenario, per-score results |
| `results/<round>/summary_*.csv`, `paired_*.csv`, `main_family_*.csv`, `secondary_*.csv`, `confirm_*.csv`, `bounds_*.csv` | generated summaries and tests |
| `results/<round>/FINDINGS.md` | findings of each round (Vietnamese) |
| `protocols/` | protocols frozen before each round (Vietnamese) |
