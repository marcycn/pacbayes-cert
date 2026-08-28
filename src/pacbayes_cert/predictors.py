"""Test-time predictors with corrected, example-weighted aggregation (fixes the
off-by-one / last-batch bugs noted in P0-6 for the predictor evaluators).

These report *descriptive deployment metrics* on the held-out test set:

* ``test_stochastic``     - single-sample Gibbs predictor (one posterior draw / example).
* ``test_posterior_mean`` - deterministic posterior-mean network.
* ``test_ensemble``       - average of ``samples`` posterior draws (finite ensemble).
* ``test_det``            - deterministic prior/reference network error.

IMPORTANT (P0-3): only the *Gibbs* predictor is covered by the formal PAC-Bayes
certificate.  Posterior-mean and ensemble test errors are reported for context and
must NOT be presented as formally certified.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def _ce_bounded_sum(outputs, target, pmin):
    return F.nll_loss(outputs, target, reduction="sum").item() / np.log(1.0 / pmin)


@torch.no_grad()
def test_stochastic(net, test_loader, pmin, device="cuda"):
    """Gibbs predictor: a fresh posterior sample is drawn per forward pass.

    Aggregation is example-weighted (sum of errors / total examples), so it is
    correct for unequal final batches.
    """
    net.eval()
    correct, ce_sum, total = 0, 0.0, 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = net(data, sample=True, clamping=True, pmin=pmin)
        ce_sum += _ce_bounded_sum(outputs, target, pmin)
        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    return ce_sum / total, 1 - correct / total


@torch.no_grad()
def test_posterior_mean(net, test_loader, pmin, device="cuda"):
    net.eval()
    correct, ce_sum, total = 0, 0.0, 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = net(data, sample=False, clamping=True, pmin=pmin)
        ce_sum += _ce_bounded_sum(outputs, target, pmin)
        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    return ce_sum / total, 1 - correct / total


@torch.no_grad()
def test_ensemble(net, test_loader, pmin, device="cuda", samples=100):
    net.eval()
    correct, ce_sum, total = 0, 0.0, 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        acc = torch.zeros(target.size(0), 10, device=device)
        for _ in range(samples):
            acc += net(data, sample=True, clamping=True, pmin=pmin).exp()
        avg = (acc / samples).clamp_min(1e-12).log()
        ce_sum += _ce_bounded_sum(avg, target, pmin)
        pred = avg.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    return ce_sum / total, 1 - correct / total


@torch.no_grad()
def test_det(net, test_loader, device="cuda"):
    """0-1 error of a deterministic network (e.g. the prior reference net)."""
    net.eval()
    correct, total = 0, 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = net(data)
        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    return 1 - correct / total
