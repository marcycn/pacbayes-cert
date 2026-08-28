#!/usr/bin/env python3
"""[SUPERSEDED -- do not run.] The figure this produced was replaced by
images/fig_partition.png, because this one showed the split as a *stratified*
subset selection. Stratification is the approach the dissertation rejects: it
sets S0's class quotas from label counts over all of S, so the prior comes to
depend on the bound set's labels. Running this again would put a figure back
into the paper that contradicts its own validity argument.

Data-partition schematic for the corrected paper (P0-2): S, S0, S\\S0 and which
predictor/estimator each feeds.  Output: results/figures/fig_partition.pdf"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")


def box(x, y, w, h, text, color):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="black", alpha=0.85))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color="black"))


box(0.2, 2.5, 2.0, 1.0, "Full train set\n(N examples)", "#cfe8ff")
box(3.0, 2.5, 2.2, 1.0, "Selected set S\n($n_{\\rm post}=|S|$)", "#bfe3c0")
arrow(2.2, 3.0, 3.0, 3.0)
ax.text(2.6, 3.4, "stratified\nperc_train", ha="center", fontsize=7)

box(6.2, 4.0, 3.4, 1.0, "$S_0$: prior subset  →  learn prior $P$", "#ffe0b3")
box(6.2, 1.5, 3.4, 1.0, "$S\\setminus S_0$: bound subset\n→ empirical Gibbs risk + certificate", "#f4c7c3")
arrow(5.2, 3.2, 6.2, 4.4)
arrow(5.2, 2.8, 6.2, 2.0)
ax.text(5.7, 3.9, "perc_prior", ha="center", fontsize=7)

ax.annotate("posterior $Q$ trained on all of $S$", (4.1, 2.4), (4.1, 0.5),
            ha="center", fontsize=8, arrowprops=dict(arrowstyle="-|>"))
ax.text(4.9, 5.4, "$n_{\\rm bound}=|S\\setminus S_0|$ enters the certificate (corrected, P0-1)",
        ha="center", fontsize=9, style="italic")

fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_partition.pdf"))
import shutil
_img = os.path.join(ROOT, "paper", "muthesis", "images")
os.makedirs(_img, exist_ok=True)
shutil.copy(os.path.join(FIG, "fig_partition.pdf"), os.path.join(_img, "fig_partition.pdf"))
print("wrote", os.path.join(FIG, "fig_partition.pdf"), "and copied to paper images")
