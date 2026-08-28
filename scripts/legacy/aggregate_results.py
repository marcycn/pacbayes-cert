#!/usr/bin/env python3
"""Aggregate PBB results: compare to JMLR 2021 Table-1 targets + produce figures.
Reads results/results_summary.csv (one row per run). Writes:
  results/comparison.md   (mine-vs-paper table + gaps)
  results/figures/cert_compare.png, slack.png, predictor_error.png
Run on the remote (has the CSV + matplotlib).  python scripts/aggregate_results.py
"""
import json, sys
from pathlib import Path
try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Need pandas+matplotlib:", e); sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "results_summary.csv"
OUT = ROOT / "results"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)

# JMLR 2021 (arXiv:2007.12911) Table-1 targets, verified from the paper PDF.
# Columns: Risk_01 (cert), Stch_01, PostMean_01, Ens_01. None = not extracted / n/a.
TARGETS = {
    "mnist_fquad_rand_fcn":    dict(risk_01=0.3155, stch=0.0951, postmean=0.0558, ens=0.0572),
    "mnist_flamb_rand_fcn":    dict(risk_01=0.3275, stch=0.0742, postmean=0.0429, ens=0.0448),
    "mnist_fclassic_rand_fcn": dict(risk_01=0.3304, stch=0.1531, postmean=0.0851, ens=0.0868),
    "mnist_bbb_rand_fcn":      dict(risk_01=0.5516, stch=None,   postmean=None,   ens=None),
    "mnist_fquad_learnt_fcn":  dict(risk_01=0.0279, stch=0.0204, postmean=0.0186, ens=0.0189),
    "mnist_flamb_learnt_fcn":  dict(risk_01=0.0354, stch=0.0178, postmean=0.0185, ens=0.0185),
    "mnist_fclassic_learnt_fcn": dict(risk_01=0.0284, stch=None, postmean=None, ens=None),
    "mnist_fquad_learnt_cnn":  dict(risk_01=0.0155, stch=0.0127, postmean=0.0105, ens=0.0104),
}

df = pd.read_csv(CSV)
df = df.sort_values("run_id").reset_index(drop=True)

# --- comparison table ---
rows = []
for _, r in df.iterrows():
    t = TARGETS.get(r["run_id"].rsplit("_seed", 1)[0], {})
    def gap(mine, key):
        pv = t.get(key)
        return f"{mine:.4f} vs {pv:.4f} (Δ{mine-pv:+.4f})" if pv is not None else f"{mine:.4f} (no target)"
    rows.append(dict(run=r["run_id"],
                     risk_01=gap(r["risk_01"], "risk_01"),
                     stch=gap(r["stch_01"], "stch"),
                     postmean=gap(r["postmean_01"], "postmean"),
                     ens=gap(r["ens_01"], "ens")))
cmp = pd.DataFrame(rows)
# write markdown by hand (avoid tabulate dependency)
def to_md(frame):
    cols = list(frame.columns)
    out = "| " + " | ".join(cols) + " |\n"
    out += "|" + "|".join(["---"] * len(cols)) + "|\n"
    for _, r in frame.iterrows():
        out += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return out
md = to_md(cmp)
(OUT / "comparison.md").write_text("# MNIST reproduction — ours vs Pérez-Ortiz 2021 (Table 1)\n\n" + md)
print(md)

# --- figures ---
d = df.copy()
d["slack"] = d["risk_01"] - d["stch_01"]
labels = [x.replace("_seed0", "").replace("mnist_", "") for x in d["run_id"]]

# 1) certificate (Risk_01) vs paper target
plt.figure(figsize=(11, 5))
x = range(len(d)); w = 0.4
mine = list(d["risk_01"]); paper = [TARGETS.get(r, {}).get("risk_01") for r in d["run_id"]]
plt.bar([i - w/2 for i in x], mine, w, label="ours (mc=10k)")
plt.bar([i + w/2 for i in x], [p if p else 0 for p in paper], w, label="paper (mc=150k)")
plt.xticks(list(x), labels, rotation=30, ha="right"); plt.ylabel("Risk certificate (0-1)"); plt.yscale("log")
plt.title("MNIST: PAC-Bayes risk certificate — ours vs Pérez-Ortiz 2021"); plt.legend(); plt.tight_layout()
plt.savefig(FIG / "cert_compare.png", dpi=150); plt.close()

# 2) certificate slack = cert - test error (tightness gap)
plt.figure(figsize=(11, 5))
plt.bar(x, list(d["slack"])); plt.xticks(list(x), labels, rotation=30, ha="right")
plt.ylabel("Slack = cert − stochastic test error"); plt.title("MNIST: certificate tightness (smaller = tighter)")
plt.tight_layout(); plt.savefig(FIG / "slack.png", dpi=150); plt.close()

# 3) predictor-rule test error (stochastic / posterior-mean / ensemble)
plt.figure(figsize=(11, 5))
plt.bar([i - w/2 for i in x], list(d["stch_01"]), w, label="stochastic")
plt.bar([i for i in x], list(d["postmean_01"]), w, label="posterior-mean")
plt.bar([i + w/2 for i in x], list(d["ens_01"]), w, label="ensemble")
plt.xticks(list(x), labels, rotation=30, ha="right"); plt.ylabel("Test error (0-1)")
plt.title("MNIST: predictor-rule test error"); plt.legend(); plt.tight_layout()
plt.savefig(FIG / "predictor_error.png", dpi=150); plt.close()
print(f"\nWrote {OUT/'comparison.md'} + 3 figures to {FIG}")
