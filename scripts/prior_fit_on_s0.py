"""Recompute the prior network's 0--1 error on its own training subset S0.

The convergence screen that decides whether a learnt-prior run is usable
originally read ``prior_net_test_01``, the prior network's error on the *test*
set.  That is outside both S0 and S_b, so it does not touch the certificate's
validity, but it is still a held-out outcome measure, and screening runs on an
outcome measure is the shape of a result filter even when it is not used as one.
The diagnostic we actually want -- "did this prior network train at all?" -- is
available without looking at any held-out data: the prior net's error on S0, the
subset it was trained on.  A prior stuck on the chance-level plateau has ~0.9
error there; one that trained has near zero.

This script recomputes that number for every learnt-prior run from the saved
``prior.pt`` and ``split_indices.npz``, so no retraining is needed, and writes it
back into ``metrics.json`` as ``prior_net_s0_01``.  Label noise is re-applied
exactly as the run applied it, since the prior for a noisy-label run was fitted
to noisy labels.

    python scripts/prior_fit_on_s0.py [--results results/raw] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pacbayes_cert import data as datamod          # noqa: E402
from pacbayes_cert.models import build_prior_net    # noqa: E402
from pacbayes_cert.predictors import test_det       # noqa: E402
from pacbayes_cert.seeds import SeedBundle          # noqa: E402


def s0_error(run_dir: str, cache: dict, device: str) -> float:
    m = json.load(open(os.path.join(run_dir, "metrics.json"), encoding="utf-8"))
    idx = np.load(os.path.join(run_dir, "split_indices.npz"))
    prior_idx = idx["prior"]
    if prior_idx.size == 0:
        raise ValueError("no S0 for %s" % run_dir)

    ds = m["dataset"]
    if ds not in cache:
        cache[ds] = datamod.load_dataset(ds, root="data")
    train, _ = cache[ds]

    # Re-apply the run's label noise the same way the run did, so the prior is
    # scored against the labels it was actually fitted to.
    if m.get("label_noise", 0.0):
        seeds = SeedBundle.from_base(m["base_seed"])
        labels = datamod.get_labels(train)
        n_classes = int(labels.max()) + 1
        train = datamod.apply_label_noise(
            train, idx["selected"], m["label_noise"], n_classes,
            seed=seeds.seed_split + 1,
        )

    subset = torch.utils.data.Subset(train, [int(i) for i in prior_idx])
    loader = torch.utils.data.DataLoader(subset, batch_size=1000, shuffle=False,
                                         num_workers=0)

    # Dropout is only active in training mode, and test_det calls net.eval(), so
    # the probability passed here does not affect the number; it has to match
    # only because it changes which modules the state dict expects.
    net = build_prior_net(m["model"], ds, 0.2, device)
    net.load_state_dict(torch.load(os.path.join(run_dir, "prior.pt"),
                                   map_location=device))
    return test_det(net, loader, device=device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/raw")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache: dict = {}
    done = skipped = 0
    for name in sorted(os.listdir(args.results)):
        d = os.path.join(args.results, name)
        mp = os.path.join(d, "metrics.json")
        if not os.path.isfile(mp) or not os.path.isfile(os.path.join(d, "prior.pt")):
            continue
        m = json.load(open(mp, encoding="utf-8"))
        if m.get("prior_type") != "learnt":
            continue
        if "prior_net_s0_01" in m and not args.dry_run:
            skipped += 1
            continue
        err = s0_error(d, cache, device)
        print("%-46s S0 %.4f   test %.4f" % (name, err, m.get("prior_net_test_01", -1)))
        if not args.dry_run:
            m["prior_net_s0_01"] = round(float(err), 6)
            with open(mp, "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2)
        done += 1
    print("\n%d evaluated, %d already had the field" % (done, skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
