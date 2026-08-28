#!/usr/bin/env python3
"""Produce the 3 dedicated report figures (values from results_summary.csv / report tables).
Run on the remote (has matplotlib). -> results/figures/{ladder,smalldata,prior}.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
FIG = Path("/root/autodl-tmp/pacbayes-cert/results/figures"); FIG.mkdir(parents=True, exist_ok=True)
C = "#1f77b4"; C2 = "#d62728"

# Fig 1 — cross-dataset certificate ladder (fquad, learnt prior)
ds = ["MNIST\n(FCN)", "Fashion-MNIST\n(FCN)", "CIFAR-10\n(9-layer CNN)"]
cert = [0.039, 0.145, 0.410]
plt.figure(figsize=(6.5, 4.2))
bars = plt.bar(ds, cert, color=C)
for b, v in zip(bars, cert): plt.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontsize=10)
plt.ylabel("0–1 risk certificate (learnt prior, fquad)"); plt.ylim(0, 0.46)
plt.title("Figure 1. Certificate tightness degrades with task difficulty")
plt.tight_layout(); plt.savefig(FIG/"ladder.png", dpi=150); plt.close()

# Fig 2 — small-data curve (MNIST fquad/learnt)
pt = [1.0, 0.5, 0.2, 0.1]; cert2 = [0.039, 0.052, 0.063, 0.072]; err2 = [0.022, 0.028, 0.048, 0.064]
plt.figure(figsize=(6.5, 4.2))
plt.plot(pt, cert2, "o-", color=C, label="Risk certificate")
plt.plot(pt, err2, "s--", color=C2, label="Stochastic test error")
plt.gca().invert_xaxis()  # 1.0 -> 0.1 left-to-right
plt.xlabel("Training-data fraction"); plt.ylabel("Rate")
for x, y in zip(pt, cert2): plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(5, 6), fontsize=9, color=C)
plt.title("Figure 2. Self-certified bound stays non-vacuous down to 10% data")
plt.legend(); plt.tight_layout(); plt.savefig(FIG/"smalldata.png", dpi=150); plt.close()

# Fig 3 — prior comparison (fquad FCN): rand vs learnt
d3 = ["MNIST", "Fashion-MNIST"]; rand = [0.327, 0.429]; learnt = [0.039, 0.145]
import numpy as np
x = np.arange(len(d3)); w = 0.38
plt.figure(figsize=(6.5, 4.2))
b1 = plt.bar(x-w/2, rand, w, color="#ff7f0e", label="data-free (rand)")
b2 = plt.bar(x+w/2, learnt, w, color=C, label="data-dependent (learnt)")
for b, v in zip(list(b1)+list(b2), rand+learnt): plt.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontsize=9)
plt.xticks(x, d3); plt.ylabel("0–1 risk certificate (fquad FCN)"); plt.yscale("log")
plt.title("Figure 3. The learnt prior is the dominant lever on tightness")
plt.legend(); plt.tight_layout(); plt.savefig(FIG/"prior.png", dpi=150); plt.close()
print("Wrote ladder.png, smalldata.png, prior.png to", FIG)
