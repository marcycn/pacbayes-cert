#!/usr/bin/env python3
"""Per-class error and confusion analysis for a trained posterior (P1-10).

Loads a saved posterior checkpoint, evaluates the posterior-mean predictor on the
test set, and writes a confusion matrix + per-class error figure. This is a
descriptive deployment-side analysis (the certified object is still the Gibbs
classifier); it shows where the error concentrates on the harder datasets.
Output: results/figures/fig_confusion_<dataset>.pdf and a per-class CSV.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from pacbayes_cert.config import ExpConfig  # noqa: E402
from pacbayes_cert.data import load_dataset  # noqa: E402
from pacbayes_cert.models import build_prior_net, build_prob_net  # noqa: E402

FASHION = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
           "Sneaker", "Bag", "Ankle boot"]
CIFAR = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MNIST = [str(i) for i in range(10)]
NAMES = {"mnist": MNIST, "fashion-mnist": FASHION, "cifar10": CIFAR}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-root", default="data_all")
    args = ap.parse_args()

    import yaml
    cfg = ExpConfig(**yaml.safe_load(open(os.path.join(args.run_dir, "config.resolved.yaml"))))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, test = load_dataset(cfg.dataset, root=args.data_root)
    rho_prior = math.log(math.exp(cfg.sigma_prior) - 1.0)
    prior_net = build_prior_net(cfg.model, cfg.dataset, 0.0, device)
    net = build_prob_net(cfg.model, cfg.dataset, rho_prior, cfg.prior_dist, device, prior_net)
    net.load_state_dict(torch.load(os.path.join(args.run_dir, "posterior.pt"), map_location=device))
    net.eval()

    loader = torch.utils.data.DataLoader(test, batch_size=500, shuffle=False)
    K = 10
    conf = np.zeros((K, K), dtype=int)
    for data, target in loader:
        data = data.to(device)
        out = net(data, sample=False, clamping=True, pmin=cfg.pmin)
        pred = out.argmax(1).cpu().numpy()
        for t, p in zip(target.numpy(), pred):
            conf[t, p] += 1

    names = NAMES[cfg.dataset]
    per_class_err = 1 - np.diag(conf) / conf.sum(1)
    figdir = os.path.join(ROOT, "results", "figures")
    os.makedirs(figdir, exist_ok=True)

    # confusion matrix (row-normalised)
    cn = conf / conf.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(K)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{cfg.dataset} posterior-mean confusion (row-normalised)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    out = os.path.join(figdir, f"fig_confusion_{cfg.dataset}.pdf")
    fig.savefig(out); plt.close(fig)

    with open(os.path.join(args.run_dir, "per_class_error.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class", "error"])
        for n, e in zip(names, per_class_err):
            w.writerow([n, round(float(e), 4)])
    # copy to paper images
    import shutil
    img = os.path.join(ROOT, "paper", "muthesis", "images")
    os.makedirs(img, exist_ok=True)
    shutil.copy(out, os.path.join(img, os.path.basename(out)))
    worst = sorted(zip(names, per_class_err), key=lambda x: -x[1])[:3]
    print(f"{cfg.dataset}: overall err {per_class_err.mean():.4f}; worst classes "
          + ", ".join(f"{n} {e:.3f}" for n, e in worst))
    print("wrote", out)


if __name__ == "__main__":
    main()
