#!/usr/bin/env python3
"""Recompute every run's certificate in place from its saved Monte-Carlo estimate,
KL, n_bound and delta budget, using the current (conservative) inv_kl.

This keeps all reported certificates consistent after the inv_kl rounding fix
(returns a strict upper bracket / 1.0 vacuous fallback). Only the certificate
fields are touched; the saved mc_err/mc_ce/kl/n_bound are the source of truth, so
no retraining is needed (P2 §5.7 checkpoint/audit benefit).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from pacbayes_cert.certificates import inv_kl  # noqa: E402

RAW = os.path.join(ROOT, "results", "raw")


def recompute(d: dict) -> dict:
    m = d["mc_samples"]
    n = d["n_bound"]
    kl = d["kl"]
    dmc = d["delta_mc"]
    dpac = d["delta_pac"]
    mc_slack = np.log(2 / dmc) / m
    pac_slack = (kl + np.log((2 * np.sqrt(n)) / dpac)) / n
    for raw_key, emp_key, cert_key in [("mc_err_01", "emp_risk_01", "cert_risk_01"),
                                       ("mc_ce", "emp_risk_ce", "cert_risk_ce")]:
        emp = inv_kl(d[raw_key], mc_slack)
        d[emp_key] = emp
        d[cert_key] = inv_kl(emp, pac_slack)
    return d


def main():
    n = 0
    for mpath in glob.glob(os.path.join(RAW, "*", "metrics.json")):
        with open(mpath) as f:
            d = json.load(f)
        if "mc_err_01" not in d or d.get("n_bound", 0) <= 0:
            continue
        d = recompute(d)
        with open(mpath, "w") as f:
            json.dump(d, f, indent=2)
        n += 1
    print(f"[recompute_certs] updated {n} runs with conservative inv_kl")


if __name__ == "__main__":
    main()
