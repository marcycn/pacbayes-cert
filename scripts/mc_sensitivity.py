#!/usr/bin/env python3
"""Recompute the certificate from a *fixed* posterior checkpoint for several MC
budgets, without retraining (Phase C / P1-6).

Loads ``posterior.pt`` + ``split_indices.npz`` + ``config.resolved.yaml`` from a
run directory, rebuilds the exact bound loader, and evaluates the certificate for
each ``m`` in ``--mc-list``.  This isolates the effect of the Monte-Carlo budget
on the raw estimate, the MC-corrected empirical risk and the final certificate —
the only thing that changes is ``m`` (the posterior is identical), so any claim
about MC budget is a genuinely controlled comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from pacbayes_cert.certificates import (N_PRIOR_CANDIDATES,  # noqa: E402
                                        certificate_from_parts,
                                        compute_certificate, inv_kl)
from pacbayes_cert.config import ExpConfig  # noqa: E402
from pacbayes_cert.data import build_loaders, get_labels, load_dataset  # noqa: E402
from pacbayes_cert.models import build_prob_net, build_prior_net  # noqa: E402
from pacbayes_cert.objectives import PBObjective  # noqa: E402
from pacbayes_cert.seeds import SeedBundle, generator_for_loader  # noqa: E402
from pacbayes_cert.splits import make_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mc-list", default="1000,2000,10000,50000,150000")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--with-analytic", action="store_true",
                    help="also record the analytic curve (raw estimate held fixed, "
                         "only the slack varying) so the two can be compared in the "
                         "artefact rather than only in the text")
    args = ap.parse_args()

    import yaml
    cfg = ExpConfig(**yaml.safe_load(open(os.path.join(args.run_dir, "config.resolved.yaml"))))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = SeedBundle.from_base(cfg.base_seed)

    train, test = load_dataset(cfg.dataset, root=args.data_root)
    labels = get_labels(train)
    n_classes = int(labels.max()) + 1
    split = make_split(len(train), cfg.prior_type, cfg.perc_train, cfg.perc_prior,
                       seeds.seed_split, labels=labels, stratified=cfg.stratified)
    loaders = build_loaders(train, test, split, cfg.batch_size,
                            generator_for_loader(seeds.seed_loader))

    rho_prior = math.log(math.exp(cfg.sigma_prior) - 1.0)
    prior_net = build_prior_net(cfg.model, cfg.dataset, 0.0, device)
    prob_net = build_prob_net(cfg.model, cfg.dataset, rho_prior, cfg.prior_dist, device, prior_net)
    prob_net.load_state_dict(torch.load(os.path.join(args.run_dir, "posterior.pt"), map_location=device))
    # Populate per-layer kl_div (only set during a training-mode forward); the KL is a
    # deterministic function of the posterior/prior (mu, rho), independent of the input.
    prob_net.train()
    with torch.no_grad():
        dummy = next(iter(loaders["bound"]))[0][:2].to(device)
        prob_net(dummy, sample=True, clamping=True, pmin=cfg.pmin)
    prob_net.eval()

    pb = PBObjective(objective=cfg.objective, pmin=cfg.pmin, classes=n_classes,
                     delta=cfg.delta_train, kl_penalty=cfg.kl_penalty, device=device,
                     n_posterior=split.n_posterior, n_bound=split.n_bound)

    rows = []
    base_mc_err = None
    for m in [int(x) for x in args.mc_list.split(",")]:
        torch.manual_seed(seeds.seed_mc)
        if device == "cuda":
            torch.cuda.manual_seed_all(seeds.seed_mc)
        cert = compute_certificate(pb, prob_net, loaders["bound_mc"], mc_samples=m,
                                   delta_mc=cfg.delta_mc, delta_pac=cfg.delta_pac)
        # Report on the same convention as every other certificate in the study:
        # with the union over candidate prior scales applied (see
        # certificates.certificate_from_parts).  The as-run value is kept beside
        # it so the size of that correction stays visible here too.
        _, cert_union = certificate_from_parts(
            cert.mc_err_01, cert.kl, cert.n_bound, m,
            cfg.delta_mc, cfg.delta_pac, N_PRIOR_CANDIDATES)
        row = dict(run_id=os.path.basename(args.run_dir.rstrip("/")), mc_samples=m,
                   mc_err_01=cert.mc_err_01, emp_risk_01=cert.emp_risk_01,
                   cert_risk_01=cert_union, cert_risk_01_nounion=cert.risk_01,
                   kl=cert.kl, n_bound=cert.n_bound)
        if base_mc_err is None:
            base_mc_err = cert.mc_err_01
        if args.with_analytic:
            # The analytic counterpart: hold the raw estimate at the value measured
            # at the run's own budget and let only the finite-MC slack move with m.
            # Recording both makes the agreement between them checkable from the
            # artefact instead of only asserted in the text.
            _, analytic = certificate_from_parts(
                base_mc_err, cert.kl, cert.n_bound, m,
                cfg.delta_mc, cfg.delta_pac, N_PRIOR_CANDIDATES)
            row["cert_risk_01_analytic"] = analytic
            row["cert_abs_diff"] = abs(analytic - cert_union)
        rows.append(row)
        print("MC", m, json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()}))

    out = args.out or os.path.join(args.run_dir, "mc_sensitivity.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
