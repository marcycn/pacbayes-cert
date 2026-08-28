"""Typed result schema with explicit metric naming (fixes P2 §5.9 / P3 §6.8).

Every run writes a single ``RunResult`` as ``metrics.json``.  The field names make
the six distinct error notions unambiguous:

* ``mc_err_01``        raw Monte-Carlo 0-1 estimate on the bound subset
* ``emp_risk_01``      finite-MC corrected empirical Gibbs-risk upper bound
* ``cert_risk_01``     final PAC-Bayes 0-1 certificate on the *true Gibbs* risk
* ``test_stoch_01``    stochastic (Gibbs) predictor test error
* ``test_postmean_01`` posterior-mean (deterministic) predictor test error
* ``test_ens_01``      finite-ensemble predictor test error

The schema version is bumped whenever the meaning of a field changes so that the
aggregation layer can reject stale rows.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

SCHEMA_VERSION = "2.0"


@dataclass
class RunResult:
    # identity / provenance
    run_id: str
    schema_version: str = SCHEMA_VERSION
    dataset: str = ""
    model: str = ""           # fcn | cnn
    objective: str = ""       # fquad | flamb | fclassic | bbb
    prior_type: str = ""      # rand | learnt
    base_seed: int = 0

    # sample accounting (the P0-1 fix surfaced here)
    n_total: int = 0
    n_selected: int = 0
    n_prior: int = 0
    n_posterior: int = 0
    n_bound: int = 0
    perc_train: float = 1.0
    perc_prior: float = 0.5
    stratified: bool = True
    split_hash: str = ""

    # confidence accounting (the P0-5 fix surfaced here)
    delta_train: float = 0.025
    delta_mc: float = 0.005
    delta_pac: float = 0.005
    joint_confidence: float = 0.99
    mc_samples: int = 0

    # certificate ingredients & results
    kl: float = 0.0
    kl_over_nbound: float = 0.0
    mc_err_01: float = 0.0
    mc_ce: float = 0.0
    emp_risk_01: float = 0.0
    emp_risk_ce: float = 0.0
    cert_risk_01: float = 0.0
    cert_risk_ce: float = 0.0

    # descriptive deployment metrics (NOT certified, see P0-3)
    test_stoch_01: float = 0.0
    test_postmean_01: float = 0.0
    test_ens_01: float = 0.0
    prior_net_test_01: float = 0.0
    # Prior-net 0-1 error on S0, the subset it was trained on.  This is the
    # convergence screen aggregate.py applies: a training diagnostic that reads
    # no held-out data, so it cannot act as a filter on results.
    prior_net_s0_01: float = 0.0

    # controlled-difficulty knob (P1-3); 0.0 for standard runs
    label_noise: float = 0.0

    # bookkeeping
    train_epochs: int = 0
    prior_epochs: int = 0
    lr: float = 0.0
    lr_prior: float = 0.0
    sigma_prior: float = 0.0
    kl_penalty: float = 0.0
    duration_sec: float = 0.0
    exit_code: int = 0
    seeds: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path) -> "RunResult":
        with open(path) as f:
            d = json.load(f)
        return RunResult(**d)
