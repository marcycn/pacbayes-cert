"""Integration tests for wiring that unit tests on individual functions miss.

A correct helper that nothing calls defends nothing. Two of the corrections in
this project live in the wiring rather than in a formula, and each is tested here
against the behaviour that would be restored if the wiring were removed:

* ``scripts/aggregate.py`` has to actually pass ``N_PRIOR_CANDIDATES`` when it
  recomputes a certificate, not merely import it. The test below drives
  ``load_runs()`` over a temporary results tree, so deleting the call site fails
  it -- an earlier version asserted things about ``certificate_from_parts`` and
  the module constant instead, and would have passed with the call site gone.
* ``runner.py`` has to reseed from ``seed_eval`` before the deployment metrics,
  so that changing ``mc_samples`` cannot move a test error. Exercising that end
  to end means training a network, which is far too slow for this suite, so the
  test asserts on the *order of operations* in the source: the reseed must occur
  after the certificate and before the first deployment metric. That is weaker
  than a behavioural test, but it does fail if the two lines are deleted, which
  is the failure mode being defended against.
"""
import importlib.util
import inspect
import json
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from pacbayes_cert.certificates import (N_PRIOR_CANDIDATES,  # noqa: E402
                                        certificate_from_parts)
from pacbayes_cert.schema import SCHEMA_VERSION  # noqa: E402
from pacbayes_cert.seeds import SeedBundle  # noqa: E402


def _load_aggregate():
    spec = importlib.util.spec_from_file_location(
        "aggregate_under_test", os.path.join(ROOT, "scripts", "aggregate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MC, CE, KL, NB, M = 0.02, 0.05, 180.0, 30000, 2000
DMC, DPAC = 0.005, 0.005


def _metrics(**over):
    """A record whose stored certificate agrees with its stored ingredients,
    which is what aggregate now requires before admitting a run."""
    _, as_run = certificate_from_parts(MC, KL, NB, M, DMC, DPAC, 1)
    _, as_run_ce = certificate_from_parts(CE, KL, NB, M, DMC, DPAC, 1)
    d = dict(run_id="t0", schema_version=SCHEMA_VERSION, exit_code=0, n_bound=NB,
             cert_risk_01=as_run, cert_risk_ce=as_run_ce, mc_err_01=MC, mc_ce=CE,
             kl=KL, mc_samples=M, delta_mc=DMC, delta_pac=DPAC,
             prior_type="learnt", prior_net_s0_01=0.01, base_seed=0,
             dataset="mnist", model="fcn", objective="fquad", perc_train=1.0,
             label_noise=0.0, duration_sec=1.0)
    d.update(over)
    return d


def _write_tree(tmp, records):
    for r in records:
        d = os.path.join(tmp, r["run_id"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(r, f)


def test_aggregate_actually_applies_the_prior_selection_union():
    agg = _load_aggregate()
    rec = _metrics()
    with tempfile.TemporaryDirectory() as tmp:
        _write_tree(tmp, [rec])
        agg.RAW = tmp
        runs, registry = agg.load_runs()

    assert len(runs) == 1, registry
    got = runs[0]
    _, expected = certificate_from_parts(
        rec["mc_err_01"], rec["kl"], rec["n_bound"], rec["mc_samples"],
        rec["delta_mc"], rec["delta_pac"], N_PRIOR_CANDIDATES)
    _, asrun = certificate_from_parts(
        rec["mc_err_01"], rec["kl"], rec["n_bound"], rec["mc_samples"],
        rec["delta_mc"], rec["delta_pac"], 1)

    # the correction must cost something, or the assertion below proves nothing
    assert expected > asrun
    assert abs(got["cert_risk_01"] - expected) < 1e-12
    assert abs(got["cert_risk_01_nounion"] - asrun) < 1e-12
    assert got["n_prior_candidates"] == N_PRIOR_CANDIDATES


def test_aggregate_rejects_a_run_whose_stored_certificate_has_drifted():
    """The stored certificate must be reproducible from the stored ingredients;
    if it is not, the run is registered as inconsistent rather than used."""
    agg = _load_aggregate()
    rec = _metrics(cert_risk_01=0.5)
    with tempfile.TemporaryDirectory() as tmp:
        _write_tree(tmp, [rec])
        agg.RAW = tmp
        runs, registry = agg.load_runs()
    assert runs == []
    assert registry[0][1] == "inconsistent"


def test_aggregate_excludes_a_learnt_prior_run_with_no_s0_diagnostic():
    """Silently defaulting the missing field to 0.0 would admit an untrained
    prior as if it had converged."""
    agg = _load_aggregate()
    rec = _metrics()
    del rec["prior_net_s0_01"]
    with tempfile.TemporaryDirectory() as tmp:
        _write_tree(tmp, [rec])
        agg.RAW = tmp
        runs, registry = agg.load_runs()
    assert runs == []
    assert registry[0][1] == "unscreened" and registry[0][2] == 0


def test_aggregate_excludes_a_prior_stuck_at_chance_on_s0():
    agg = _load_aggregate()
    rec = _metrics(prior_net_s0_01=0.8991)
    with tempfile.TemporaryDirectory() as tmp:
        _write_tree(tmp, [rec])
        agg.RAW = tmp
        runs, registry = agg.load_runs()
    assert runs == []
    assert registry[0][1] == "not_converged"


def test_dedup_records_the_run_it_drops():
    agg = _load_aggregate()
    a = _metrics(run_id="a", duration_sec=10.0)
    b = _metrics(run_id="b", duration_sec=99.0)
    reg = []
    kept = agg.dedup([a, b], reg)
    assert [k["run_id"] for k in kept] == ["b"]
    assert reg and reg[0][0] == "a" and reg[0][1] == "superseded"


def test_seed_eval_is_a_distinct_stream():
    b = SeedBundle.from_base(0)
    vals = [b.seed_split, b.seed_model, b.seed_loader, b.seed_mc, b.seed_eval]
    assert len(set(vals)) == 5
    assert b.seed_eval != SeedBundle.from_base(1).seed_eval


def test_runner_reseeds_before_the_deployment_metrics():
    """Ordering test on the source: reseed after the certificate, before the
    first deployment metric. See the module docstring for why this is not a
    behavioural test."""
    from pacbayes_cert import runner

    src = inspect.getsource(runner.run_experiment)
    i_cert = src.index("compute_certificate(")
    i_seed = src.index("seeds.seed_eval")
    i_stoch = src.index("test_stochastic(")
    assert i_cert < i_seed < i_stoch, (
        "seed_eval must be applied between the certificate and the deployment "
        "metrics, or the test errors inherit the certificate's RNG state")


def test_deployment_draw_does_not_depend_on_the_mc_budget():
    """The property the reseed buys: with it, the deployment draw is the same at
    two Monte-Carlo budgets; without it, it is not."""
    b = SeedBundle.from_base(3)

    def draw(mc_samples, reseed):
        torch.manual_seed(b.seed_mc)
        torch.randn(mc_samples)             # stands in for the certificate's draws
        if reseed:
            torch.manual_seed(b.seed_eval)
        return torch.randn(4)

    assert not torch.allclose(draw(100, reseed=False), draw(200, reseed=False))
    assert torch.allclose(draw(100, reseed=True), draw(200, reseed=True))
