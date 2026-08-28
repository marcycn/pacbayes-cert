#!/usr/bin/env python3
"""Generate all paper figures from results/processed ONLY (fixes P2 §5.5).

No certificate number is ever hard-coded here; every value is read from
``results/processed/summary.csv`` / ``runs.csv`` / per-run ``mc_sensitivity.csv``.
Outputs vector PDFs to ``results/figures/``.
"""
from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "results", "processed")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def read_csv(name):
    path = os.path.join(PROC, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def f(row, key, default=0.0):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return default


def fig_certificate_anatomy(summary):
    cells = [r for r in summary if r["prior_type"] == "learnt" and f(r, "label_noise") == 0
             and abs(f(r, "perc_train") - 1.0) < 1e-9 and r["objective"] == "fquad"]
    if not cells:
        return
    cells.sort(key=lambda r: (r["dataset"], r["model"]))
    labels = [f"{r['dataset']}\n{r['model']}" for r in cells]
    emp = [f(r, "emp_risk_01_mean") for r in cells]
    cert = [f(r, "cert_risk_01_mean") for r in cells]
    slack = [c - e for c, e in zip(cert, emp)]
    x = np.arange(len(cells))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, emp, label="empirical Gibbs risk (MC-corrected)")
    ax.bar(x, slack, bottom=emp, label="PAC-Bayes complexity slack (KL + confidence)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("0–1 risk certificate")
    ax.set_title("Certificate anatomy (fquad, learnt prior, full data)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_certificate_anatomy.pdf")); plt.close(fig)


def fig_prior_comparison(summary):
    by = defaultdict(dict)
    for r in summary:
        if r["objective"] != "fquad" or f(r, "label_noise") != 0 or abs(f(r, "perc_train") - 1) > 1e-9:
            continue
        by[(r["dataset"], r["model"])][r["prior_type"]] = r
    paired = [(k, v) for k, v in by.items() if "learnt" in v and "rand" in v]
    if not paired:
        return
    paired.sort()
    labels = [f"{k[0]}\n{k[1]}" for k, _ in paired]
    learnt = [f(v["learnt"], "cert_risk_01_mean") for _, v in paired]
    learnt_sd = [f(v["learnt"], "cert_risk_01_std") for _, v in paired]
    rand = [f(v["rand"], "cert_risk_01_mean") for _, v in paired]
    rand_sd = [f(v["rand"], "cert_risk_01_std") for _, v in paired]
    x = np.arange(len(paired)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, learnt, w, yerr=learnt_sd, capsize=3, label="learnt prior")
    ax.bar(x + w / 2, rand, w, yerr=rand_sd, capsize=3, label="random prior")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("0–1 risk certificate"); ax.set_title("Learnt vs random prior (paired cells only)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_prior_comparison.pdf")); plt.close(fig)


def fig_smalldata(summary):
    cells = [r for r in summary if r["dataset"] == "mnist" and r["model"] == "fcn"
             and r["objective"] == "fquad" and r["prior_type"] == "learnt" and f(r, "label_noise") == 0]
    if len(cells) < 2:
        return
    cells.sort(key=lambda r: f(r, "perc_train"))
    pt = [f(r, "perc_train") * 100 for r in cells]
    cert = [f(r, "cert_risk_01_mean") for r in cells]
    sd = [f(r, "cert_risk_01_std") for r in cells]
    nb = [int(f(r, "n_bound_mean")) for r in cells]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(pt, cert, yerr=sd, marker="o", capsize=3)
    for xi, yi, n in zip(pt, cert, nb):
        ax.annotate(f"n_bound={n}", (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("% of training data used"); ax.set_ylabel("0–1 risk certificate")
    ax.set_title("Small-data certificate (corrected n_bound, error bars over seeds)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_smalldata.pdf")); plt.close(fig)


def fig_mc_convergence():
    files = glob.glob(os.path.join(ROOT, "results", "raw", "*", "mc_sensitivity.csv"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for path in sorted(files):
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        m = [int(r["mc_samples"]) for r in rows]
        rid = rows[0]["run_id"]
        ax.plot(m, [f(r, "mc_err_01") for r in rows], "o--", alpha=0.6, label=f"{rid} raw")
        ax.plot(m, [f(r, "emp_risk_01") for r in rows], "s-", alpha=0.6, label=f"{rid} MC-corrected")
        ax.plot(m, [f(r, "cert_risk_01") for r in rows], "^-", alpha=0.9, label=f"{rid} certificate")
    ax.set_xscale("log"); ax.set_xlabel("MC samples m"); ax.set_ylabel("0–1 risk")
    ax.set_title("MC convergence on a fixed checkpoint"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_mc_convergence.pdf")); plt.close(fig)


def fig_difficulty(summary):
    cells = [r for r in summary if r["dataset"] == "mnist" and r["model"] == "fcn"
             and r["objective"] == "fquad" and r["prior_type"] == "learnt"]
    if len(cells) < 2:
        return
    cells.sort(key=lambda r: f(r, "label_noise"))
    ln = [f(r, "label_noise") * 100 for r in cells]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, lab in [("emp_risk_01_mean", "empirical risk"), ("kl_over_nbound_mean", "KL/n_bound"),
                     ("cert_risk_01_mean", "certificate")]:
        ax.plot(ln, [f(r, key) for r in cells], "o-", label=lab)
    ax.set_xlabel("label noise (%)"); ax.set_ylabel("value")
    ax.set_title("Controlled difficulty (same arch / data size, MNIST FCN)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_difficulty.pdf")); plt.close(fig)


def _copy_to_paper():
    import shutil
    img = os.path.join(ROOT, "paper", "muthesis", "images")
    os.makedirs(img, exist_ok=True)
    for fn in os.listdir(FIG):
        if fn.startswith("fig_") and fn.endswith(".pdf"):
            shutil.copy(os.path.join(FIG, fn), os.path.join(img, fn))


def main():
    summary = read_csv("summary.csv")
    fig_certificate_anatomy(summary)
    fig_prior_comparison(summary)
    fig_smalldata(summary)
    fig_mc_convergence()
    fig_difficulty(summary)
    _copy_to_paper()
    print(f"[make_figures] wrote PDFs to {FIG} and copied to paper/muthesis/images")


if __name__ == "__main__":
    main()
