"""End-to-end experiment runner: trains a posterior and emits a certified,
fully-provenanced :class:`RunResult` plus reusable checkpoints.

Artifacts written to ``outdir``:
    config.resolved.yaml   resolved configuration
    split_indices.npz      selected / prior / bound indices + class histograms
    prior.pt               deterministic prior network state (learnt prior only)
    posterior.pt           probabilistic posterior network state
    metrics.json           the typed RunResult
    environment.json       provenance (versions, git, gpu)
    stdout.log             captured log (written by the CLI wrapper)

The split/checkpoint persistence (P2 §5.7) lets the certificate be recomputed
cheaply after fixing MC budget / delta / n_bound without retraining.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch

from . import data as datamod
from . import provenance
from .certificates import compute_certificate
from .config import ExpConfig
from .models import (Lambda_var, build_prior_net, build_prob_net, count_parameters)
from .objectives import PBObjective
from .predictors import test_det, test_ensemble, test_posterior_mean, test_stochastic
from .schema import RunResult
from .seeds import (SeedBundle, generator_for_loader, rng_for_split, seed_model_rngs,
                    set_deterministic)
from .splits import assert_loaders_match_split, make_split, split_hash


def make_run_id(cfg: ExpConfig) -> str:
    rid = f"{cfg.dataset}_{cfg.objective}_{cfg.prior_type}_{cfg.model}_seed{cfg.base_seed}"
    if cfg.perc_train != 1.0:
        rid += f"_pt{int(round(cfg.perc_train*100))}"
    if cfg.label_noise:
        rid += f"_ln{int(round(cfg.label_noise*100))}"
    return rid


def run_experiment(cfg: ExpConfig, outdir: str, data_root: str = "data", verbose=False) -> RunResult:
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    seeds = SeedBundle.from_base(cfg.base_seed)
    set_deterministic(True)

    # ---- data + explicit split (P0-1/P0-2/P1-9) -----------------------------
    train, test = datamod.load_dataset(cfg.dataset, root=data_root)
    labels = datamod.get_labels(train)
    n_classes = int(labels.max()) + 1
    split = make_split(
        n_total=len(train), prior_type=cfg.prior_type, perc_train=cfg.perc_train,
        perc_prior=cfg.perc_prior, seed_split=seeds.seed_split, labels=labels,
        stratified=cfg.stratified,
    )

    # optional controlled label noise on the training set (P1-3)
    train_for_loaders = train
    if cfg.label_noise > 0:
        train_for_loaders = datamod.apply_label_noise(
            train, np.array(split.selected_indices), cfg.label_noise, n_classes,
            seed=seeds.seed_split + 1,
        )

    loader_gen = generator_for_loader(seeds.seed_loader)
    # The dataset is materialised as tensors up front (see data._TensorCached),
    # so there is nothing for worker processes to decode; they would only add
    # inter-process overhead, and on Windows they are re-spawned every epoch.
    # Measured on this hardware: num_workers=1 took 17.7 s/epoch, 0 took 10.9 s,
    # and 0 with the tensor cache takes 1.1 s. The sampler runs in the main
    # process either way, so the batch order is unaffected.
    loader_kwargs = {"num_workers": 0}
    loaders = datamod.build_loaders(
        train_for_loaders, test, split, cfg.batch_size, loader_gen, loader_kwargs=loader_kwargs,
        mc_eval_batch=cfg.mc_eval_batch,
    )
    assert_loaders_match_split(loaders, split)

    # ---- model init & training (seed_model controls all torch/np RNGs) ------
    seed_model_rngs(seeds.seed_model)
    rho_prior = math.log(math.exp(cfg.sigma_prior) - 1.0)
    dropout = 0.0 if cfg.prior_type == "rand" else cfg.dropout_prob

    prior_net = build_prior_net(cfg.model, cfg.dataset, dropout, device)
    prior_net_s0_01 = 0.0
    if cfg.prior_type == "learnt":
        from .trainer import train_prior_net
        train_prior_net(prior_net, loaders["prior"], cfg.prior_epochs, cfg.lr_prior,
                        cfg.momentum_prior, device=device, verbose=verbose)
        # The convergence screen of scripts/aggregate.py reads this: the prior
        # net's error on the subset it was just trained on.  Scoring it on S0
        # rather than on the test set keeps the screen a training diagnostic --
        # it consults no held-out data and exists before any certificate does,
        # so it cannot be turned into a filter on results.
        prior_net_s0_01 = test_det(prior_net, loaders["prior"], device=device)
    prior_net_test_01 = test_det(prior_net, loaders["test"], device=device)

    prob_net = build_prob_net(cfg.model, cfg.dataset, rho_prior, cfg.prior_dist, device, prior_net)

    pbobj = PBObjective(
        objective=cfg.objective, pmin=cfg.pmin, classes=n_classes, delta=cfg.delta_train,
        kl_penalty=cfg.kl_penalty, device=device,
        n_posterior=split.n_posterior, n_bound=split.n_bound,
    )

    lambda_var = None
    if cfg.objective == "flamb":
        lambda_var = Lambda_var(cfg.initial_lamb, split.n_posterior).to(device)

    from .trainer import train_posterior
    train_posterior(prob_net, pbobj, loaders["posterior"], cfg.train_epochs, cfg.lr, cfg.momentum,
                    lambda_var=lambda_var, device=device, verbose=verbose)

    # ---- certificate (seed_mc controls MC weight sampling; P0-5/P0-6) -------
    torch.manual_seed(seeds.seed_mc)
    if device == "cuda":
        torch.cuda.manual_seed_all(seeds.seed_mc)
    cert = compute_certificate(
        pbobj, prob_net, loaders["bound_mc"], mc_samples=cfg.mc_samples,
        delta_mc=cfg.delta_mc, delta_pac=cfg.delta_pac,
    )

    # ---- descriptive deployment metrics (NOT certified; P0-3) ---------------
    # Reseed from seed_eval rather than letting these continue the certificate's
    # RNG state: otherwise the stochastic and ensemble test errors depend on
    # mc_samples and on the bound-set batching, so changing the Monte-Carlo
    # budget would silently move a test error.
    torch.manual_seed(seeds.seed_eval)
    if device == "cuda":
        torch.cuda.manual_seed_all(seeds.seed_eval)
    _, test_stoch_01 = test_stochastic(prob_net, loaders["test"], cfg.pmin, device=device)
    _, test_postmean_01 = test_posterior_mean(prob_net, loaders["test"], cfg.pmin, device=device)
    _, test_ens_01 = test_ensemble(prob_net, loaders["test"], cfg.pmin, device=device,
                                   samples=cfg.samples_ensemble)

    dt = round(time.time() - t0, 1)

    # ---- persist artifacts --------------------------------------------------
    np.savez(
        os.path.join(outdir, "split_indices.npz"),
        selected=np.array(split.selected_indices),
        prior=np.array(split.prior_indices),
        bound=np.array(split.bound_indices),
        prior_class_hist=np.array(split.prior_class_hist),
        bound_class_hist=np.array(split.bound_class_hist),
    )
    torch.save(prob_net.state_dict(), os.path.join(outdir, "posterior.pt"))
    if cfg.prior_type == "learnt":
        torch.save(prior_net.state_dict(), os.path.join(outdir, "prior.pt"))
    try:
        import yaml
        with open(os.path.join(outdir, "config.resolved.yaml"), "w") as f:
            yaml.safe_dump(cfg.to_dict(), f, sort_keys=True)
    except Exception:
        pass

    prov = provenance.collect(cfg.to_dict())
    prov["n_params"] = count_parameters(prob_net)
    import json
    with open(os.path.join(outdir, "environment.json"), "w") as f:
        json.dump(prov, f, indent=2)

    result = RunResult(
        run_id=make_run_id(cfg), dataset=cfg.dataset, model=cfg.model, objective=cfg.objective,
        prior_type=cfg.prior_type, base_seed=cfg.base_seed,
        n_total=split.n_total, n_selected=split.n_selected, n_prior=split.n_prior,
        n_posterior=split.n_posterior, n_bound=split.n_bound, perc_train=cfg.perc_train,
        perc_prior=cfg.perc_prior, stratified=split.stratified, split_hash=split_hash(split),
        delta_train=cfg.delta_train, delta_mc=cfg.delta_mc, delta_pac=cfg.delta_pac,
        joint_confidence=cert.joint_confidence, mc_samples=cfg.mc_samples,
        kl=cert.kl, kl_over_nbound=cert.kl_over_nbound,
        mc_err_01=cert.mc_err_01, mc_ce=cert.mc_ce, emp_risk_01=cert.emp_risk_01,
        emp_risk_ce=cert.emp_risk_ce, cert_risk_01=cert.risk_01, cert_risk_ce=cert.risk_ce,
        test_stoch_01=test_stoch_01, test_postmean_01=test_postmean_01, test_ens_01=test_ens_01,
        prior_net_test_01=prior_net_test_01, prior_net_s0_01=prior_net_s0_01,
        label_noise=cfg.label_noise,
        train_epochs=cfg.train_epochs, prior_epochs=cfg.prior_epochs if cfg.prior_type == "learnt" else 0,
        lr=cfg.lr, lr_prior=cfg.lr_prior, sigma_prior=cfg.sigma_prior, kl_penalty=cfg.kl_penalty,
        duration_sec=dt, exit_code=0, seeds=seeds.to_dict(), provenance=prov,
    )
    result.to_json(os.path.join(outdir, "metrics.json"))
    return result
