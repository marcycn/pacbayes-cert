"""Recompute every saved data partition and check it against the one stored.

The dissertation's central validity claim is that no reported run used a
label-dependent split. A configuration flag cannot establish that: fifteen
data-free-prior runs still carry ``stratified: true`` from an older config
generation, and ``make_split`` ignores the flag rather than honouring it, so the
flag says nothing about what actually happened.

What does establish it is recomputing each partition from its recorded seed with
the current label-independent code and comparing index-for-index against the
``split_indices.npz`` the run saved. If they match for every run, then every
reported certificate was evaluated on a bound set drawn without reference to any
label, whatever the config happened to say.

    python scripts/verify_splits.py [--results results/raw]

Exit status is non-zero if any run disagrees, so this can gate a release.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pacbayes_cert import data as datamod      # noqa: E402
from pacbayes_cert.seeds import SeedBundle     # noqa: E402
from pacbayes_cert.splits import make_split    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/raw")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    labels_for: dict = {}
    checked = 0
    mismatches = []
    flagged = []

    for mpath in sorted(glob.glob(os.path.join(args.results, "*", "metrics.json"))):
        run_dir = os.path.dirname(mpath)
        npz_path = os.path.join(run_dir, "split_indices.npz")
        if not os.path.isfile(npz_path):
            continue
        d = json.load(io.open(mpath, encoding="utf-8"))

        ds = d["dataset"]
        if ds not in labels_for:
            train, _ = datamod.load_dataset(ds, root=args.data_root)
            labels_for[ds] = datamod.get_labels(train)
        labels = labels_for[ds]

        seeds = SeedBundle.from_base(d["base_seed"])
        sp = make_split(n_total=len(labels), prior_type=d["prior_type"],
                        perc_train=d["perc_train"], perc_prior=d["perc_prior"],
                        seed_split=seeds.seed_split, labels=labels)

        z = np.load(npz_path)
        same = (np.array_equal(np.sort(z["bound"]), np.sort(np.asarray(sp.bound_indices)))
                and np.array_equal(np.sort(z["prior"]), np.sort(np.asarray(sp.prior_indices)))
                and np.array_equal(np.sort(z["selected"]), np.sort(np.asarray(sp.selected_indices))))
        checked += 1
        if not same:
            mismatches.append(d["run_id"])
        if d.get("stratified"):
            flagged.append((d["run_id"], d["prior_type"]))

    print(f"[verify_splits] {checked} partitions recomputed, {len(mismatches)} mismatched")
    if flagged and not args.quiet:
        learnt = [r for r, p in flagged if p == "learnt"]
        print(f"[verify_splits] {len(flagged)} runs carry stratified=true in their config "
              f"({len(learnt)} of them learnt-prior); the flag is inert, and the "
              f"recomputation above is what shows the split was label-independent")
        if learnt:
            print("[verify_splits] WARNING: a learnt-prior run requested stratification")
            return 2
    for r in mismatches:
        print("   MISMATCH", r)
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
