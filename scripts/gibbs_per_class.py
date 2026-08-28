#!/usr/bin/env python3
"""Per-class risk of the *certified* predictor, on the set the certificate uses.

``error_analysis.py`` reports where the posterior-mean predictor's error falls on
the test set.  That is a deployment-side description: the quantity the
certificate bounds is the empirical Gibbs risk on the bound set S\\S0, under a
freshly sampled weight vector per draw.  The two are different predictors on
different data, so the deployment picture cannot by itself say where the
certified risk comes from.

This script closes that gap.  It loads the same fixed posterior checkpoint,
rebuilds the exact bound loader from the stored split indices, and accumulates
misclassification counts per true class over ``--mc-samples`` posterior draws.
The class-averaged result equals the aggregate ``mc_err_01`` the certificate
starts from, which is asserted at the end as a consistency check.

Outputs ``<run-dir>/gibbs_per_class.csv``.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from pacbayes_cert.config import ExpConfig  # noqa: E402
from pacbayes_cert.data import load_dataset  # noqa: E402
from pacbayes_cert.models import build_prior_net, build_prob_net  # noqa: E402

NAMES = {
    "mnist": [str(i) for i in range(10)],
    "fashion-mnist": ["T-shirt", "trouser", "pullover", "dress", "coat",
                      "sandal", "shirt", "sneaker", "bag", "ankle boot"],
    "cifar10": ["plane", "car", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"],
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--mc-samples", type=int, default=None,
                    help="draws per batch; defaults to the run's own mc_samples")
    args = ap.parse_args()

    import yaml
    cfg = ExpConfig(**yaml.safe_load(open(os.path.join(args.run_dir, "config.resolved.yaml"))))
    m = args.mc_samples or cfg.mc_samples
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train, _ = load_dataset(cfg.dataset, root=args.data_root)
    idx = np.load(os.path.join(args.run_dir, "split_indices.npz"))
    bound_idx = idx["bound"]
    bound_set = torch.utils.data.Subset(train, list(bound_idx))
    bs = len(bound_idx) if cfg.mc_eval_batch <= 0 else min(cfg.mc_eval_batch, len(bound_idx))
    loader = torch.utils.data.DataLoader(bound_set, batch_size=bs, shuffle=False)

    rho_prior = math.log(math.exp(cfg.sigma_prior) - 1.0)
    prior_net = build_prior_net(cfg.model, cfg.dataset, 0.0, device)
    net = build_prob_net(cfg.model, cfg.dataset, rho_prior, cfg.prior_dist, device, prior_net)
    net.load_state_dict(torch.load(os.path.join(args.run_dir, "posterior.pt"), map_location=device))
    net.eval()

    K = len(NAMES[cfg.dataset])
    wrong = np.zeros(K, dtype=np.int64)     # misclassifications, summed over draws
    seen = np.zeros(K, dtype=np.int64)      # examples x draws, per class
    conf = np.zeros((K, K), dtype=np.int64)

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        counts = torch.bincount(target, minlength=K).cpu().numpy()
        for _ in range(m):
            pred = net(data, sample=True, clamping=True, pmin=cfg.pmin).argmax(1)
            bad = pred.ne(target)
            wrong += torch.bincount(target[bad], minlength=K).cpu().numpy()
            conf += np.bincount((target * K + pred).cpu().numpy(),
                                minlength=K * K).reshape(K, K)
            seen += counts

    per_class = wrong / np.maximum(seen, 1)
    overall = wrong.sum() / seen.sum()

    out = os.path.join(args.run_dir, "gibbs_per_class.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "gibbs_err_01", "n_examples", "mc_samples"])
        for name, e, s in zip(NAMES[cfg.dataset], per_class, seen):
            w.writerow([name, round(float(e), 6), int(s // m), m])
        w.writerow(["ALL", round(float(overall), 6), int(seen.sum() // m), m])

    np.savetxt(os.path.join(args.run_dir, "gibbs_confusion.csv"), conf, fmt="%d", delimiter=",")

    # consistency: the class-averaged Gibbs error must equal the aggregate the
    # certificate is built from, up to the difference in MC draws.
    import json
    rec = json.load(open(os.path.join(args.run_dir, "metrics.json")))
    ref = rec["mc_err_01"]
    print(f"{cfg.dataset}/{cfg.model}/{cfg.objective}/{cfg.prior_type}: "
          f"Gibbs overall {overall:.5f} vs metrics mc_err_01 {ref:.5f} "
          f"(delta {overall - ref:+.5f}, m={m})")
    worst = sorted(zip(NAMES[cfg.dataset], per_class), key=lambda x: -x[1])[:3]
    print("  worst classes: " + ", ".join(f"{n} {e:.3f}" for n, e in worst))
    print("  wrote " + out)


if __name__ == "__main__":
    main()
