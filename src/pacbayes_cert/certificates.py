"""PAC-Bayes-kl risk certificate with corrected Monte-Carlo aggregation and
explicit joint-confidence accounting (fixes P0-5 and P0-6).

The certificate is a double inversion of the binary KL:

1. *Finite-MC step.*  The Gibbs risk on the bound set is estimated by ``m`` Monte-
   Carlo draws of the posterior weights.  Inverting the binary KL with slack
   ``ln(2/delta_mc)/m`` turns the empirical MC mean into a high-probability upper
   bound ``hat R`` on the *true* Gibbs risk over the bound sample, valid with prob
   >= 1 - delta_mc.
2. *PAC-Bayes step.*  Inverting the binary KL again with slack
   ``(KL + ln(2 sqrt(n_bound)/delta_pac)) / n_bound`` turns ``hat R`` into a bound
   on the population Gibbs risk, valid with prob >= 1 - delta_pac.

By a union bound the final certificate holds with probability at least
``1 - delta_mc - delta_pac``.  The original code used a single ``delta_test`` for
*both* steps and then reported "99% confidence", which is really only
``1 - 2*delta_test`` (P0-5).

Monte-Carlo aggregation (P0-6): losses are accumulated *example-weighted* over the
bound set and averaged over exactly ``m`` MC samples (the original code divided by
``batch_id``, an off-by-one, and equal-weighted unequal final batches). For the
headline FCN cells the bound set is evaluated as a single batch, so the estimate is
exactly the mean of ``m`` i.i.d. whole-set Gibbs draws, which is the estimator the
finite-MC concentration step assumes. The example-weighted aggregation is exact and
batching-independent for a fixed set of sampled weights (verified for a deterministic
posterior in the tests); for a stochastic posterior different batchings draw
different weights, so they agree only up to Monte-Carlo noise, not bit-for-bit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


def _binary_kl(qs: float, p: float) -> float:
    """Binary KL ``kl(qs || p)`` with the standard 0/1 edge conventions."""
    if p <= 0.0 or p >= 1.0:
        return float("inf")
    if qs <= 0.0:
        return -math.log(1.0 - p)
    if qs >= 1.0:
        return -math.log(p)
    return qs * math.log(qs / p) + (1.0 - qs) * math.log((1.0 - qs) / (1.0 - p))


def inv_kl(qs: float, ks: float) -> float:
    """Conservative inverse of the binary KL.

    Returns an *upper* value ``p >= qs`` such that ``kl(qs || p) <= ks`` is the
    binding constraint, rounding towards the larger (safer) side so the result is
    a valid upper bound on the risk. If even the numerical cap satisfies the
    constraint (a numerically vacuous bound) it returns ``1.0``.

    This fixes two issues a strict certificate must avoid: the previous version
    returned the last bisection midpoint (which can fall just below the true
    inverse) and capped at ``1-1e-10`` without a vacuous fallback.
    """
    if ks <= 0.0:
        return qs
    lo = qs
    hi = 1.0 - 1e-12
    if _binary_kl(qs, hi) <= ks:
        return 1.0  # bound is (numerically) vacuous -> round up to 1
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _binary_kl(qs, mid) <= ks:
            lo = mid
        else:
            hi = mid
    return hi  # smallest p that violates the constraint => conservative upper bound


@torch.no_grad()
def mc_sampling(pbobj, net, bound_loader, mc_samples: int, clamping=True):
    """Example-weighted, MC-averaged empirical Gibbs risk on the bound set.

    Returns ``(ce_mean, err_mean)`` where both are averages over the whole bound
    set and over ``mc_samples`` posterior draws.  Correct aggregation (P0-6):

      * 0-1 error: accumulate misclassification *counts* over examples and MC draws,
        divide by ``n_examples * mc_samples`` -> exact, batching-invariant.
      * cross-entropy: accumulate the per-example bounded NLL summed over examples
        and MC draws, divide by ``n_examples * mc_samples``.
    """
    net.eval()
    device = pbobj.device
    total_examples = 0
    err_count = 0.0   # summed over examples AND mc draws
    ce_sum = 0.0      # summed over examples AND mc draws (bounded NLL)
    log_inv_pmin = np.log(1.0 / pbobj.pmin)

    for data, target in bound_loader:
        data, target = data.to(device), target.to(device)
        batch_n = target.size(0)
        total_examples += batch_n
        for _ in range(mc_samples):
            outputs = net(data, sample=True, clamping=clamping, pmin=pbobj.pmin)
            # bounded NLL summed over the batch (nll_loss(reduction='sum') / log(1/pmin))
            ce_batch = torch.nn.functional.nll_loss(outputs, target, reduction="sum").item()
            ce_sum += ce_batch / log_inv_pmin
            pred = outputs.max(1, keepdim=True)[1]
            err_count += (pred.ne(target.view_as(pred))).sum().item()

    denom = total_examples * mc_samples
    return ce_sum / denom, err_count / denom


@dataclass
class Certificate:
    """All quantities entering and resulting from the certificate computation."""

    # raw MC estimates on the bound set
    mc_err_01: float
    mc_ce: float
    # finite-MC corrected empirical upper bounds
    emp_risk_01: float
    emp_risk_ce: float
    # final population Gibbs-risk certificates
    risk_01: float
    risk_ce: float
    # ingredients
    kl: float
    kl_over_nbound: float
    n_bound: int
    mc_samples: int
    delta_mc: float
    delta_pac: float
    joint_confidence: float

    def to_dict(self) -> dict:
        return {
            "mc_err_01": self.mc_err_01,
            "mc_ce": self.mc_ce,
            "emp_risk_01": self.emp_risk_01,
            "emp_risk_ce": self.emp_risk_ce,
            "risk_01": self.risk_01,
            "risk_ce": self.risk_ce,
            "kl": self.kl,
            "kl_over_nbound": self.kl_over_nbound,
            "n_bound": self.n_bound,
            "mc_samples": self.mc_samples,
            "delta_mc": self.delta_mc,
            "delta_pac": self.delta_pac,
            "joint_confidence": self.joint_confidence,
        }


#: Size of the data-independent grid the prior scale is unioned over.
#:
#: The tempting number here is 3, or 4: sigma_0 was settled by comparing
#: certificates at {0.01, 0.02, 0.05} and then adopting 0.03.  But 0.03 was
#: picked *after* seeing those three curves, so "the candidate set" is not a set
#: of four fixed in advance -- any value in the flat region could have been
#: adopted instead, and a union over a set assembled after looking at S_b buys
#: nothing.  We therefore union over a wider grid instead: every three-decimal
#: scale in (0, 1), i.e. 0.001 through 0.999, which we believe is a superset of
#: the values the probe could have returned.
#:
#: What this does NOT establish: the grid was written down after the fact, so we
#: cannot claim the selection procedure was confined to it in advance.  The union
#: is valid for any grid fixed independently of S_b and covers the adopted scale
#: however it was picked from that grid; the residual exposure is the gap between
#: "we believe the procedure could only have returned a three-decimal scale" and
#: a pre-registered candidate set.  The paper carries this as a threat to
#: validity rather than as a solved problem, and so should any reader of this
#: constant.
#: The price is log(999) = 6.91 nats instead of log(4) = 1.39, which on the
#: smallest bound set in this study (n_b = 3,000) costs 0.004 of certificate and
#: on the headline cells about 0.0005.
N_PRIOR_CANDIDATES = 999


def certificate_from_parts(mc_err: float, kl: float, n_bound: int,
                           mc_samples: int, delta_mc: float, delta_pac: float,
                           n_prior_candidates: int = 1) -> tuple:
    """Recompute a certificate from saved ingredients, with an optional union
    over candidate priors.

    Returns ``(emp_risk, cert_risk)``.

    ``n_prior_candidates`` exists because the prior scale sigma_0 was not fixed
    a priori: it was chosen after comparing certificates, and those certificates
    were computed on the bound set.  PAC-Bayes-kl holds for a prior fixed
    independently of S_b, simultaneously over all posteriors -- it does not
    licence looking at S_b and then picking one prior for free.  The repair is a
    union bound over a candidate grid: apply the inequality to each of the K
    priors at ``delta_pac / K`` and the guarantee holds simultaneously for all of
    them, so whichever is adopted is covered, at a cost of ``log K`` in the
    numerator of the PAC-Bayes slack.  For this to mean anything the grid has to
    be independent of the data -- see :data:`N_PRIOR_CANDIDATES` for why that
    rules out simply counting the values we happened to try.

    Nothing analogous is needed for the posterior.  ``Q`` may depend on all of
    ``S``, including ``S_b``, because the inequality is already simultaneous over
    posteriors; only the prior has to be pinned down in advance.

    Passing ``1`` reproduces the uncorrected value a run recorded, which is what
    the equivalence test in the suite checks.
    """
    emp = inv_kl(mc_err, np.log(2 / delta_mc) / mc_samples)
    delta_eff = delta_pac / n_prior_candidates
    slack = (kl + np.log((2 * np.sqrt(n_bound)) / delta_eff)) / n_bound
    return emp, inv_kl(emp, slack)


def compute_certificate(
    pbobj,
    net,
    bound_loader,
    mc_samples: int,
    delta_mc: float,
    delta_pac: float,
    precomputed_mc=None,
) -> Certificate:
    """Compute the PAC-Bayes-kl certificate with split confidence budget.

    ``precomputed_mc`` may be ``(mc_ce, mc_err_01)`` to recompute the certificate
    from a fixed checkpoint's MC estimates without re-sampling (used by the MC
    sensitivity sweep and by recomputation after fixing n_bound).
    """
    kl = float(net.compute_kl().item())
    n_bound = pbobj.n_bound

    if precomputed_mc is not None:
        mc_ce, mc_err_01 = precomputed_mc
    else:
        mc_ce, mc_err_01 = mc_sampling(pbobj, net, bound_loader, mc_samples)

    # finite-MC inversion (delta_mc)
    mc_slack = np.log(2 / delta_mc) / mc_samples
    emp_risk_ce = inv_kl(mc_ce, mc_slack)
    emp_risk_01 = inv_kl(mc_err_01, mc_slack)

    # PAC-Bayes inversion (delta_pac)
    pac_slack = (kl + np.log((2 * np.sqrt(n_bound)) / delta_pac)) / n_bound
    risk_ce = inv_kl(emp_risk_ce, pac_slack)
    risk_01 = inv_kl(emp_risk_01, pac_slack)

    return Certificate(
        mc_err_01=mc_err_01,
        mc_ce=mc_ce,
        emp_risk_01=emp_risk_01,
        emp_risk_ce=emp_risk_ce,
        risk_01=risk_01,
        risk_ce=risk_ce,
        kl=kl,
        kl_over_nbound=kl / n_bound,
        n_bound=n_bound,
        mc_samples=mc_samples,
        delta_mc=delta_mc,
        delta_pac=delta_pac,
        joint_confidence=1.0 - delta_mc - delta_pac,
    )
