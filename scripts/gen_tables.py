#!/usr/bin/env python3
"""Generate LaTeX result tables from results/processed/summary.csv (fixes P2 §5.6).

The paper does ``\\input{generated/main_table.tex}`` so numbers can never drift from
the registry.  Columns follow the transparent naming required by P3 §6.8/§6.9:
n_bound, MC m, empirical Gibbs risk, KL, KL/n_bound, certificate, and the three
*descriptive* test errors (Gibbs / posterior-mean / ensemble).
"""
from __future__ import annotations

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "results", "processed")
OUT = os.path.join(ROOT, "paper", "muthesis", "generated")
os.makedirs(OUT, exist_ok=True)


def read_summary():
    p = os.path.join(PROC, "summary.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def g(r, k, d=0.0):
    try:
        return float(r[k])
    except Exception:
        return d


def ms(r, key):
    return f"${g(r, key+'_mean'):.4f}\\pm{g(r, key+'_std'):.4f}$"


# ---- presentation labels -------------------------------------------------
# The registry stores internal configuration keys.  Tables in the dissertation
# must use the notation defined in the text, never the raw config strings.
_DATASET = {"mnist": "MNIST", "fashion-mnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}
_MODEL = {"fcn": "FCN", "cnn": "CNN"}
_PRIOR = {"learnt": "learnt", "rand": "data-free"}
_OBJECTIVE = {
    "fquad": r"$f_{\text{quad}}$",
    "flamb": r"$f_{\lambda}$",
    "fclassic": r"$f_{\text{classic}}$",
    "bbb": r"$f_{\text{bbb}}$",
}


def lab(mapping, key):
    return mapping.get(key, key)


def condition(r):
    """Human-readable experimental condition for a per-run row."""
    pt = float(r.get("perc_train", 1) or 1)
    ln = float(r.get("label_noise", 0) or 0)
    if ln > 0:
        return f"noise {int(round(ln * 100))}\\%"
    if abs(pt - 1.0) > 1e-9:
        return f"data {int(round(pt * 100))}\\%"
    return "headline"


def main_table(rows):
    rows = [r for r in rows if float(r["perc_train"]) == 1.0 and float(r["label_noise"]) == 0]
    rows.sort(key=lambda r: (r["dataset"], r["model"], r["prior_type"], r["objective"]))
    lines = [
        r"\begin{tabular}{llll r r r r r r}",
        r"\toprule",
        r"Dataset & Model & Prior & Obj. & $n_{\text{bound}}$ & Emp.\ Gibbs & KL$/n_{\text{bound}}$ "
        r"& \textbf{Certificate} & Gibbs test & Post.mean test \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{lab(_DATASET, r['dataset'])} & {lab(_MODEL, r['model'])} & "
            f"{lab(_PRIOR, r['prior_type'])} & {lab(_OBJECTIVE, r['objective'])} & "
            f"{int(g(r,'n_bound_mean'))} & {ms(r,'emp_risk_01')} & ${g(r,'kl_over_nbound_mean'):.5f}$ & "
            f"{ms(r,'cert_risk_01')} & {ms(r,'test_stoch_01')} & {ms(r,'test_postmean_01')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    # 10 columns: the paper sets this landscape (sidewaystable) so it can be read
    # at the body font size instead of being scaled down to fit the text width.
    with open(os.path.join(OUT, "main_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def smalldata_table(rows):
    rows = [r for r in rows if r["dataset"] == "mnist" and r["model"] == "fcn"
            and r["objective"] == "fquad" and r["prior_type"] == "learnt" and float(r["label_noise"]) == 0]
    if not rows:
        return
    rows.sort(key=lambda r: -float(r["perc_train"]))
    lines = [r"\begin{tabular}{r r r r r}", r"\toprule",
             r"\% data & $n_{\text{posterior}}$ & $n_{\text{bound}}$ & Emp.\ Gibbs & \textbf{Certificate} \\",
             r"\midrule"]
    for r in rows:
        pct = int(round(float(r["perc_train"]) * 100))
        nb = int(g(r, "n_bound_mean"))
        npost = nb * 2  # perc_prior=0.5
        lines.append(f"{pct}\\% & {npost} & {nb} & {ms(r,'emp_risk_01')} & {ms(r,'cert_risk_01')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT, "smalldata_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


_MACRO_MAP = str.maketrans({"-": "", "_": "", "0": "Zero", "1": "One", "2": "Two",
                            "3": "Three", "4": "Four", "5": "Five", "6": "Six",
                            "7": "Seven", "8": "Eight", "9": "Nine", ".": "p"})


def _macro(*parts):
    return "".join(p.translate(_MACRO_MAP).capitalize() for p in parts if p)


def perseed_table():
    """Per-seed appendix table read from results/processed/runs.csv."""
    p = os.path.join(PROC, "runs.csv")
    if not os.path.exists(p):
        return
    with open(p) as f:
        runs = list(csv.DictReader(f))
    runs.sort(key=lambda r: (r["dataset"], r["model"], r["prior_type"], r["objective"],
                             float(r.get("perc_train", 1)), float(r.get("label_noise", 0)),
                             int(r["base_seed"])))
    header = (r"Dataset & Model & Prior & Obj. & Condition & Seed & $n_{\text{b}}$ "
              r"& Cert & Stoch & PM & Ens \\")
    # The longtable carries a caption so that it takes a table number of its own;
    # without one it still advances the counter and leaves a gap in the sequence.
    lines = [r"{\small\begin{longtable}[]{@{}lllllr r r r r r@{}}",
             # The count is computed, not written down: a hard-coded one goes
             # stale the moment a run is added and then disagrees with the rows
             # printed underneath it.
             rf"\caption{{Per-seed results for all {len(runs)} reported runs, "
             r"one row per run.}"
             r"\label{tab:perseed}\\",
             r"\toprule\noalign{}",
             header,
             r"\midrule\noalign{}\endfirsthead",
             r"\multicolumn{11}{@{}l}{\emph{Table \thetable\ continued from previous page}}\\",
             r"\toprule\noalign{}",
             header,
             r"\midrule\noalign{}\endhead",
             r"\midrule\multicolumn{11}{r@{}}{\emph{continued on next page}}\\\endfoot",
             r"\bottomrule\noalign{}\endlastfoot"]
    prev_group = None
    for r in runs:
        # blank rule between experiment families so the table is self-describing
        group = (r["dataset"], r["model"], r["prior_type"], r["objective"], condition(r))
        if prev_group is not None and group != prev_group:
            lines.append(r"\addlinespace[2pt]")
        prev_group = group
        lines.append(
            f"{lab(_DATASET, r['dataset'])} & {lab(_MODEL, r['model'])} & "
            f"{lab(_PRIOR, r['prior_type'])} & {lab(_OBJECTIVE, r['objective'])} & "
            f"{condition(r)} & "
            f"{r['base_seed']} & {int(g(r,'n_bound'))} & {g(r,'cert_risk_01'):.4f} & "
            f"{g(r,'test_stoch_01'):.4f} & {g(r,'test_postmean_01'):.4f} & {g(r,'test_ens_01'):.4f} \\\\")
    lines += [r"\end{longtable}}"]
    # longtable cannot be wrapped in resizebox; use scriptsize + tighter columns instead.
    text = "\n".join(lines).replace(r"{\small\begin{longtable}", r"{\scriptsize\begin{longtable}")
    with open(os.path.join(OUT, "perseed_table.tex"), "w") as f:
        f.write(text + "\n")


def reproduction_table(rows):
    """Published-vs-ours comparison for the cells this project re-ran.

    The published values live in a read-only reference file that records which
    source column each number came from, so the comparison can never silently
    drift onto a different column of the reference table (P0-4).
    """
    ref_path = os.path.join(ROOT, "results", "reference", "perezortiz2021_table1.csv")
    if not os.path.exists(ref_path):
        return
    with open(ref_path) as f:
        ref = list(csv.DictReader(line for line in f if not line.startswith("#")))
    index = {(r["dataset"], r["model"], r["objective"], r["prior_type"]): r
             for r in rows if float(r["perc_train"]) == 1.0 and float(r["label_noise"]) == 0}

    dstats = {"cert": [], "stoch": []}
    lines = [r"\begin{tabular}{@{}lll rr r rr r@{}}", r"\toprule",
             r"& & & \multicolumn{3}{c}{Certificate} & \multicolumn{3}{c}{Stochastic test error} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r"Model & Prior & Obj. & Published & Ours & $\Delta$ & Published & Ours & $\Delta$ \\",
             r"\midrule"]
    for rr in ref:
        key = (rr["dataset"], rr["model"], rr["objective"], rr["prior_type"])
        r = index.get(key)
        if r is None:
            continue
        rc, rs = float(rr["ref_cert_risk_01"]), float(rr["ref_test_stoch_01"])
        oc, os_ = g(r, "cert_risk_01_mean"), g(r, "test_stoch_01_mean")
        dstats["cert"].append(oc - rc)
        dstats["stoch"].append(os_ - rs)
        lines.append(
            f"{lab(_MODEL, r['model'])} & {lab(_PRIOR, r['prior_type'])} & "
            f"{lab(_OBJECTIVE, r['objective'])} & "
            f"${rc:.4f}$ & ${oc:.4f}$ & ${oc-rc:+.4f}$ & "
            f"${rs:.4f}$ & ${os_:.4f}$ & ${os_-rs:+.4f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT, "reproduction_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # The spread of the reproduction deltas is quoted in the prose. Deriving it
    # here rather than typing it keeps it from going stale when the runs change,
    # which is exactly how an earlier revision came to quote a range that no
    # longer matched the table above it.
    return dstats


def repro_range_macros(dstats):
    """Macros for the spread of the reproduction deltas quoted in the prose."""
    if not dstats or not dstats.get("stoch"):
        return []
    out = []
    for key, name in (("stoch", "Stoch"), ("cert", "Cert")):
        v = dstats[key]
        out.append(r"\newcommand{\Repro%sDeltaMin}{%+.4f}" % (name, min(v)))
        out.append(r"\newcommand{\Repro%sDeltaMax}{%+.4f}" % (name, max(v)))
        out.append(r"\newcommand{\Repro%sDeltaAbsMax}{%.4f}" % (name, max(abs(x) for x in v)))
    return out


def numbers_tex(rows, dstats=None):
    """Emit \\newcommand macros so inline numbers in the prose come from the registry."""
    lines = ["% AUTO-GENERATED from results/processed/summary.csv -- do not edit.",
             "% Usage: \\certMnistFcnFquadLearnt etc. (mean, 4 dp).",
             "\\providecommand{\\nresult}[1]{#1}"]
    for r in rows:
        key = _macro(r["dataset"], r["model"], r["objective"], r["prior_type"])
        pt = float(r["perc_train"])
        ln = float(r["label_noise"])
        # LaTeX control sequences must be letters only -> translate digits to words.
        if abs(pt - 1.0) > 1e-9:
            key += "Pt" + str(int(round(pt * 100))).translate(_MACRO_MAP)
        if ln > 0:
            key += "Ln" + str(int(round(ln * 100))).translate(_MACRO_MAP)
        for metric, fld in [("Cert", "cert_risk_01_mean"), ("Stoch", "test_stoch_01_mean"),
                            ("Postmean", "test_postmean_01_mean"), ("Kl", "kl_mean"),
                            ("Nbound", "n_bound_mean"), ("Emp", "emp_risk_01_mean")]:
            val = g(r, fld)
            sval = str(int(round(val))) if metric == "Nbound" else f"{val:.4f}"
            lines.append(f"\\newcommand{{\\{metric}{key}}}{{{sval}}}")
    lines += repro_range_macros(dstats)
    with open(os.path.join(OUT, "numbers.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    rows = read_summary()
    if not rows:
        print("[gen_tables] no summary.csv yet; run aggregate.py after experiments")
        # still emit placeholders so LaTeX compiles
        for name in ("main_table.tex", "smalldata_table.tex"):
            p = os.path.join(OUT, name)
            if not os.path.exists(p):
                open(p, "w").write("\\emph{(table pending experiment results)}\n")
        if not os.path.exists(os.path.join(OUT, "numbers.tex")):
            open(os.path.join(OUT, "numbers.tex"), "w").write("% pending results\n")
        return
    main_table(rows)
    smalldata_table(rows)
    dstats = reproduction_table(rows)
    perseed_table()
    numbers_tex(rows, dstats)
    print(f"[gen_tables] wrote tables + numbers to {OUT}")


if __name__ == "__main__":
    main()
