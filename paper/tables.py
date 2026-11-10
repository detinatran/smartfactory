"""Tables 1-4 of the paper -> outputs/tables/{mech,uses,frozen,paired_hier}.tex.
Run main_tests.py and hier_bootstrap.py first (Table 4 reads their outputs)."""
import pandas as pd

from common import DATASETS as DS, DNAME as DN, FAULTS, OUT, per_seed, read, write_tex

# ---- Table 1: mechanism on identical clean/faulty windows (results/mechanism_summary.csv, from scripts/fault_mechanism.py)
g = read("mechanism_summary.csv").set_index(["kind", "dataset"]).loc["Unseen faults"]
L = ["\\begin{table}[t]", "\\centering", "\\footnotesize", "\\begin{tabular}{lcccccc}", "\\hline",
     " & $|\\Delta|/$ & $\\rho(\\Delta,$ & $\\rho(\\Delta,$ & \\multicolumn{3}{c}{Fault-induced errors rejected}\\\\",
     "Data & spread & $\\Delta$spread) & $\\Delta$one-sided) & one-sided & spread & MC dropout\\\\", "\\hline"]
for d in DS:
    r = g.loc[d]
    L.append(f"{DN[d]} & {r.shift_spread:.1f} & {r.rho_spread:+.2f} & {r.rho_asym:+.2f} & "
             f"{100 * r.rej_asym:.0f}\\% & {100 * r.rej_std:.0f}\\% & {100 * r.rej_mcd:.0f}\\%\\\\")
L += ["\\hline", "\\end{tabular}",
      "\\caption{Six unseen faults on identical windows with and without the fault. $\\Delta$: prediction change; $\\rho$: Spearman correlation with the change of spread or one-sided score; last columns: share of fault-induced optimistic errors rejected at 90\\% coverage (random: 10\\%). GRU/TCN, 10 seeds.}",
      "\\label{tab:mech}", "\\end{table}"]
write_tex("mech.tex", L)

# ---- Table 2: what the score uses, incl. fault detectors (CPU replay, results/replay_scores_per_fold.csv.gz)
rp = read("replay_scores_per_fold.csv.gz")
x = per_seed(rp[rp.scenario.isin(FAULTS)], ["risk_matched", "reject_new"]).groupby(["dataset", "score"]).mean() * 100
rows = [("ensemble_std", "Ensemble spread"), ("fault_detector", "Fault detector, window statistics"),
        ("fault_detector_full", "Fault detector, all features"), ("pred_only", "One-sided, prediction only"),
        ("pred_cycle_only", "One-sided, prediction + elapsed time"), ("full", "One-sided, all features"),
        ("one_sided_plus_detector", "\\quad + fault-detector output")]
L = ["\\begin{table}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{lcccc}", "\\hline",
     "Score & " + " & ".join(DN[d] for d in DS) + "\\\\", "\\hline"]
for k, name in rows:
    L.append(f"{name} & " + " & ".join(f"{x.loc[(d, k), 'risk_matched']:.1f} ({x.loc[(d, k), 'reject_new']:.0f})" for d in DS) + "\\\\")
L += ["\\hline", "\\end{tabular}",
      "\\caption{Risk (\\%) at 90\\% matched coverage on unseen faults (brackets: \\% of fault-induced optimistic errors rejected). Scores re-fitted on selector units (fixed hyperparameters) on CPU-replayed predictions of the same checkpoints.}",
      "\\label{tab:uses}", "\\end{table}"]
write_tex("uses.tex", L)

# ---- Table 3: frozen thresholds (results/frozen_per_seed.csv.gz)
M = {"asymmetric_risk": "One-sided (ours)", "cqr_lower": "CQR lower bound", "ensemble_std": "Ensemble spread",
     "mc_dropout": "MC dropout", "deep_ens_nll": "Deep ens.\\ (NLL)", "selectivenet": "SelectiveNet"}
d = read("frozen_per_seed.csv.gz")
d = d[d.model.isin(["gru", "tcn"]) & d.method.isin(M)]
clean = d[d.scenario == "clean"].groupby(["dataset", "method"])[["coverage", "optimistic_error_rate"]].mean()
fault = d[d.scenario.isin(FAULTS)].groupby(["dataset", "method", "model", "seed", "scenario"])[["coverage", "optimistic_error_rate"]] \
    .mean().groupby(["dataset", "method"]).mean()
t = pd.DataFrame({"C_clean": 100 * clean.coverage, "C_fault": 100 * fault.coverage, "R_fault": 100 * fault.optimistic_error_rate})
t["dC"] = t.C_fault - t.C_clean
t.to_csv(OUT / "stats/table3_coverage_shift.csv")
L = ["\\begin{table}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
     "\\begin{tabular}{l" + "c" * len(DS) + "}", "\\hline", "Selector & " + " & ".join(DN[s] for s in DS) + "\\\\", "\\hline"]
for m, name in M.items():
    cells = []
    for s in DS:
        dc = int(round(t.loc[(s, m), "dC"]))
        cells.append(f"{t.loc[(s, m), 'R_fault']:.1f} / {t.loc[(s, m), 'C_fault']:.0f} ({f'{dc:+d}' if dc else '0'})")
    L.append(f"{name} & " + " & ".join(cells) + "\\\\")
L += ["\\hline", "\\end{tabular}",
      "\\caption{Thresholds frozen on clean calibration units (90\\% coverage), unseen faults: risk \\% / coverage \\% ($\\Delta C$: vs.\\ clean, points).}",
      "\\label{tab:frozen}", "\\end{table}"]
write_tex("frozen.tex", L)

# ---- Table 4: paired differences with hierarchical intervals and Holm stars
hb = pd.read_csv(OUT / "stats/hierarchical_bootstrap.csv")
st = pd.read_csv(OUT / "stats/main_family.csv")
st = st[(st.setting == "fault") & (st.metric == "risk90")]
names = [("ensemble_std", "Ensemble spread"), ("mc_dropout", "MC dropout"), ("deep_ens_nll", "Deep ens.\\ (NLL)"),
         ("selectivenet", "SelectiveNet"), ("symmetric_risk", "Symmetric event classifier"), ("cqr_lower", "CQR lower bound"),
         ("aug_vs_asym", "Ours + fault augm.")]
L = ["\\begin{table}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{2pt}", "\\resizebox{\\textwidth}{!}{%",
     "\\begin{tabular}{lcccc}", "\\hline", "Comparator & " + " & ".join(DN[s] for s in DS) + "\\\\", "\\hline"]
for k, name in names:
    cells = []
    for s in DS:
        h = hb[(hb.dataset == s) & (hb.comparator == k)].iloc[0]
        star = "$^*$" if st[(st.dataset == s) & (st.comparator == k)].iloc[0].p_holm < 0.05 else ""
        cells.append(f"{h['mean']:+.2f} [{h.lo:+.2f},{h.hi:+.2f}]{star}")
    L.append(f"{name} & " + " & ".join(cells) + "\\\\")
L += ["\\hline", "\\end{tabular}}",
      "\\caption{Unseen faults: risk at 90\\% coverage, ours minus comparator (points; negative favours ours; last row: ours also trained on the other five faults minus ours). Brackets: 95\\% hierarchical bootstrap; $^*$: Holm-adjusted Wilcoxon $p<0.05$, 20 pairs.}",
      "\\label{tab:paired}", "\\end{table}"]
write_tex("paired_hier.tex", L)
print("tables written to", OUT / "tables")
