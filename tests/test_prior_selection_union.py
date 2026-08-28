"""The prior scale sigma_0 was chosen by comparing certificates computed on the
bound set, so the reported certificate carries a union over the candidate priors.

These tests pin the three properties the correction has to have: it reduces to
the as-run value at K=1, it is conservative for K>1, and it costs exactly
log(K)/n_bound in the PAC-Bayes slack rather than something larger.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pacbayes_cert.certificates import (N_PRIOR_CANDIDATES,  # noqa: E402
                                        certificate_from_parts, inv_kl)

# ingredients of a real run: mnist_fquad_learnt_fcn_seed0
MC, KL, NB, M = 0.019834, 179.1369171142578, 30000, 2000
DMC, DPAC = 0.005, 0.005


def test_k1_reproduces_the_uncorrected_certificate():
    emp, cert = certificate_from_parts(MC, KL, NB, M, DMC, DPAC, 1)
    expected_emp = inv_kl(MC, math.log(2 / DMC) / M)
    expected = inv_kl(expected_emp, (KL + math.log(2 * math.sqrt(NB) / DPAC)) / NB)
    assert abs(emp - expected_emp) < 1e-12
    assert abs(cert - expected) < 1e-12


def test_union_is_conservative_and_monotone_in_k():
    certs = [certificate_from_parts(MC, KL, NB, M, DMC, DPAC, k)[1]
             for k in (1, 2, 4, 8)]
    assert all(b >= a - 1e-15 for a, b in zip(certs, certs[1:]))
    assert certs[-1] > certs[0]


def test_union_does_not_touch_the_monte_carlo_step():
    # the correction is on the PAC-Bayes failure probability only; the empirical
    # risk that comes out of the finite-MC inversion must be identical
    e1, _ = certificate_from_parts(MC, KL, NB, M, DMC, DPAC, 1)
    e4, _ = certificate_from_parts(MC, KL, NB, M, DMC, DPAC, 4)
    assert e1 == e4


def test_cost_is_exactly_log_k_in_the_slack():
    # k is fixed here rather than read from N_PRIOR_CANDIDATES: at K=1 the
    # log-K term vanishes and the assertion would hold for any implementation,
    # so a test parameterised by the constant would pass even if the constant
    # were silently reset to 1.
    emp, _ = certificate_from_parts(MC, KL, NB, M, DMC, DPAC, 1)
    base = (KL + math.log(2 * math.sqrt(NB) / DPAC)) / NB
    for k in (2, 4, 16):
        _, cert = certificate_from_parts(MC, KL, NB, M, DMC, DPAC, k)
        assert abs(cert - inv_kl(emp, base + math.log(k) / NB)) < 1e-12


def test_the_grid_is_the_data_independent_one_not_the_values_we_tried():
    """The certificates are covered only if the union runs over a grid fixed
    without reference to the data.  Counting the scales we happened to compare
    ({0.01, 0.02, 0.05} plus the adopted 0.03, so K=4) would be a set assembled
    after looking at the bound set, and would not cover the selection; the grid
    is every three-decimal scale in (0, 1) instead.  This pins the constant so
    that reverting it to 4, or to 1, fails here rather than silently weakening
    every reported certificate."""
    assert N_PRIOR_CANDIDATES == 999
