#!/usr/bin/env python3
"""Monte-Carlo budget curve for a fixed checkpoint, computed analytically (P1-6).

For a fixed posterior the only way the Monte-Carlo budget m enters the certificate
is through the finite-MC slack ln(2/delta_mc)/m of the first KL inversion: the raw
MC mean is an unbiased estimate of the Gibbs risk, so we hold it at its best
available value (the headline run's estimate) and vary only m. This isolates the
*systematic* effect of the sample budget --- the MC-correction shrinking as m grows
--- from sampling noise, and lets us show the curve up to the reference's
m=150,000 without re-sampling. Output: <run-dir>/mc_sensitivity.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from pacbayes_cert.certificates import inv_kl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mc-list", default="1000,2000,5000,10000,50000,150000")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = json.load(open(os.path.join(args.run_dir, "metrics.json")))
    raw = d["mc_err_01"]
    kl = d["kl"]
    n = d["n_bound"]
    dmc, dpac = d["delta_mc"], d["delta_pac"]
    pac_slack = (kl + np.log((2 * np.sqrt(n)) / dpac)) / n

    rows = []
    for m in [int(x) for x in args.mc_list.split(",")]:
        emp = inv_kl(raw, np.log(2 / dmc) / m)
        cert = inv_kl(emp, pac_slack)
        rows.append(dict(run_id=os.path.basename(args.run_dir.rstrip("/")), mc_samples=m,
                         mc_err_01=raw, emp_risk_01=emp, cert_risk_01=cert, kl=kl, n_bound=n))
        print("MC", m, "emp", round(emp, 5), "cert", round(cert, 5))

    out = args.out or os.path.join(args.run_dir, "mc_sensitivity.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
