#!/usr/bin/env python3
"""Build the processed result registry from raw per-run metrics.json files.

Fixes P2 §5.5/§5.6/§5.10:
  * every figure/table downstream reads ONLY from results/processed/*, never a
    hand-typed number;
  * runs are de-duplicated by the full composite key (dataset, model, objective,
    prior_type, perc_train, label_noise, base_seed);
  * an explicit inclusion/exclusion registry records why any run is dropped;
  * schema_version is checked and stale rows are rejected.

Outputs:
  results/processed/runs.csv        one row per included run (full schema)
  results/processed/summary.csv     one row per cell: mean/std across seeds
  results/processed/registry.csv    run_id | status | included_in_main | reason
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from pacbayes_cert.schema import SCHEMA_VERSION  # noqa: E402
from pacbayes_cert.certificates import (  # noqa: E402
    N_PRIOR_CANDIDATES, certificate_from_parts)

RAW = os.path.join(ROOT, "results", "raw")
PROC = os.path.join(ROOT, "results", "processed")

CLASSES = 10
CHANCE_LEVEL = 1.0 - 1.0 / CLASSES   # 0-1 error of a predictor that has learnt nothing

CELL_KEY = ["dataset", "model", "objective", "prior_type", "perc_train", "label_noise"]
AGG_FIELDS = ["cert_risk_01", "cert_risk_ce", "cert_risk_01_nounion", "emp_risk_01",
              "kl", "kl_over_nbound", "mc_err_01", "test_stoch_01",
              "test_postmean_01", "test_ens_01", "n_bound"]


def _mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def load_runs():
    runs, registry = [], []
    for mpath in sorted(glob.glob(os.path.join(RAW, "*", "metrics.json"))):
        run_dir = os.path.dirname(mpath)
        run_name = os.path.basename(run_dir)
        try:
            with open(mpath) as f:
                d = json.load(f)
        except Exception as e:
            registry.append((run_name, "unreadable", 0, f"json error: {e}"))
            continue
        if d.get("schema_version") != SCHEMA_VERSION:
            registry.append((run_name, "stale_schema", 0,
                             f"schema {d.get('schema_version')} != {SCHEMA_VERSION}"))
            continue
        if d.get("exit_code", 0) != 0:
            registry.append((run_name, "failed", 0, f"exit_code {d.get('exit_code')}"))
            continue
        if not (d.get("n_bound", 0) > 0 and math.isfinite(d.get("cert_risk_01", float("nan")))):
            registry.append((run_name, "invalid", 0, "n_bound<=0 or non-finite certificate"))
            continue
        # A learnt prior whose deterministic network never left the chance-level
        # plateau is not an instance of the protocol we mean to report: the
        # posterior is then initialised at an untrained prior, so the cell says
        # nothing about learnt priors.
        #
        # The screen reads the prior net's error on S0, the subset it was
        # trained on, and so is a training diagnostic: it looks at no held-out
        # data at all, and it is available before any certificate or test error
        # is computed.  An earlier version screened on the prior net's *test*
        # error, which gives the same verdict on every run in this study but is
        # the wrong shape of criterion -- discarding runs on a held-out outcome
        # measure is what a result filter looks like, whatever the intent.
        # Chance is 1 - 1/CLASSES; we allow five percentage points of slack.
        # Separation is wide: trained priors reach at most 0.2281 on S0 and the
        # two failures sit at 0.8659 and 0.8991.  Data-free priors are exempt because
        # they train no prior network at all.
        if d.get("prior_type") == "learnt":
            s0 = d.get("prior_net_s0_01")
            if s0 is None:
                registry.append((run_name, "unscreened", 0,
                                 "no prior_net_s0_01: run scripts/prior_fit_on_s0.py"))
                continue
            if s0 >= CHANCE_LEVEL - 0.05:
                registry.append((run_name, "not_converged", 0,
                                 "learnt prior did not train: prior-net error on "
                                 f"S0 is {s0:.4f}, at chance"))
                continue

        # Reported certificates carry a union over the candidate prior scales
        # (see certificates.certificate_from_parts).  Both columns are recomputed
        # from the stored ingredients rather than copied from the run: copying
        # the recorded value into the "nounion" column would silently mislabel it
        # if a future run ever recorded an already-corrected certificate.
        parts = (d["kl"], d["n_bound"], d["mc_samples"], d["delta_mc"], d["delta_pac"])
        _, cert = certificate_from_parts(d["mc_err_01"], *parts, N_PRIOR_CANDIDATES)
        _, cert_ce = certificate_from_parts(d["mc_ce"], *parts, N_PRIOR_CANDIDATES)
        _, cert_nounion = certificate_from_parts(d["mc_err_01"], *parts, 1)
        _, cert_ce_nounion = certificate_from_parts(d["mc_ce"], *parts, 1)

        # The recomputation must reproduce what the run itself recorded.  A
        # mismatch means the stored ingredients and the stored certificate have
        # drifted apart, which would invalidate every downstream number, so it is
        # a hard failure rather than a warning.
        drift = abs(cert_nounion - d["cert_risk_01"])
        if drift > 1e-9:
            registry.append((run_name, "inconsistent", 0,
                             f"stored certificate and stored ingredients disagree by {drift:.2e}"))
            continue

        d["cert_risk_01_nounion"] = cert_nounion
        d["cert_risk_ce_nounion"] = cert_ce_nounion
        d["cert_risk_01"] = cert
        d["cert_risk_ce"] = cert_ce
        d["n_prior_candidates"] = N_PRIOR_CANDIDATES

        runs.append(d)
        registry.append((run_name, "ok", 1, ""))
    return runs, registry


def dedup(runs, registry=None):
    """Keep one run per (cell-key + base_seed) and record the rest as superseded.

    A cell should have exactly one run per seed; a second appears only when a
    cell has been re-run without the old directory being cleared.  Where that
    happens we keep the *longest* run, on the reasoning that a truncated or
    crashed attempt is the one to discard, and we mark the loser in the registry
    rather than dropping it silently -- an earlier version returned the survivors
    with no record that anything had been removed, which is precisely the kind of
    invisible filtering this registry exists to prevent.

    On the reported study this is inert: all 115 run directories carry distinct
    keys, so nothing is superseded.
    """
    best = {}
    for d in runs:
        key = tuple(d[k] for k in CELL_KEY) + (d["base_seed"],)
        prev = best.get(key)
        if prev is None or d.get("duration_sec", 0) > prev.get("duration_sec", 0):
            if prev is not None and registry is not None:
                registry.append((prev["run_id"], "superseded", 0,
                                 "another run shares this cell key and seed and ran longer"))
            best[key] = d
        elif registry is not None:
            registry.append((d["run_id"], "superseded", 0,
                             "another run shares this cell key and seed and ran longer"))
    return list(best.values())


def write_runs(runs):
    os.makedirs(PROC, exist_ok=True)
    if not runs:
        return
    fields = list(runs[0].keys())
    fields = [f for f in fields if f not in ("seeds", "provenance")]
    with open(os.path.join(PROC, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for d in sorted(runs, key=lambda r: r["run_id"]):
            w.writerow(d)


def write_summary(runs):
    cells = {}
    for d in runs:
        key = tuple(d[k] for k in CELL_KEY)
        cells.setdefault(key, []).append(d)
    rows = []
    for key, ds in sorted(cells.items()):
        row = dict(zip(CELL_KEY, key))
        row["n_seeds"] = len(ds)
        row["seeds"] = "|".join(str(x["base_seed"]) for x in sorted(ds, key=lambda r: r["base_seed"]))
        for fld in AGG_FIELDS:
            m, s = _mean_std([x[fld] for x in ds])
            row[f"{fld}_mean"] = round(m, 6)
            row[f"{fld}_std"] = round(s, 6)
        rows.append(row)
    if not rows:
        return
    with open(os.path.join(PROC, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def write_registry(registry):
    os.makedirs(PROC, exist_ok=True)
    with open(os.path.join(PROC, "registry.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "status", "included_in_main", "exclusion_reason"])
        w.writerows(sorted(registry))


def main():
    runs, registry = load_runs()
    superseded = []
    runs = dedup(runs, superseded)
    # A run marked superseded was admitted by load_runs, so its earlier "ok" row
    # has to be replaced rather than merely appended to.
    dropped = {r[0] for r in superseded}
    registry = [r for r in registry if r[0] not in dropped] + superseded
    write_runs(runs)
    summary = write_summary(runs) or []
    write_registry(registry)
    n_inc = sum(r[2] for r in registry)
    print(f"[aggregate] {len(runs)} runs included, {len(registry)-n_inc} excluded, "
          f"{len(summary)} cells -> {PROC}")


if __name__ == "__main__":
    main()
