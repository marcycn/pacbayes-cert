"""P0-6 / DoD: Monte-Carlo aggregation is invariant to batching.

A model with a *deterministic* posterior (zero std) removes MC noise so one-batch
and multi-batch estimates must match exactly.  A second check uses a stochastic
posterior with a fixed seed and an unequal final batch, requiring agreement within
a tight tolerance.
"""
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pacbayes_cert.certificates import mc_sampling
from pacbayes_cert.models import ProbNNet4l
from pacbayes_cert.objectives import PBObjective


def _deterministic_net():
    # rho very negative -> sigma ~ 0 -> sampling == posterior mean (deterministic)
    net = ProbNNet4l(rho_prior=-30.0, device="cpu")
    net.eval()
    return net


def _make_loaders(n=130, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    ds = TensorDataset(x, y)
    one = DataLoader(ds, batch_size=n, shuffle=False)
    multi = DataLoader(ds, batch_size=32, shuffle=False)  # 130 = 32*4 + 2 (unequal last)
    return one, multi


def test_one_vs_multi_batch_deterministic():
    net = _deterministic_net()
    pb = PBObjective(objective="fquad", pmin=1e-5, classes=10, device="cpu")
    one, multi = _make_loaders()
    ce1, er1 = mc_sampling(pb, net, one, mc_samples=3)
    ce2, er2 = mc_sampling(pb, net, multi, mc_samples=3)
    assert abs(er1 - er2) < 1e-9
    assert abs(ce1 - ce2) < 1e-6


def test_one_vs_multi_batch_stochastic_seeded():
    pb = PBObjective(objective="fquad", pmin=1e-5, classes=10, device="cpu")
    one, multi = _make_loaders()
    net = ProbNNet4l(rho_prior=-3.0, device="cpu")
    net.eval()
    torch.manual_seed(123)
    ce1, er1 = mc_sampling(pb, net, one, mc_samples=200)
    torch.manual_seed(123)
    ce2, er2 = mc_sampling(pb, net, multi, mc_samples=200)
    # different batching changes the sequence of random draws, but the
    # example-weighted mean over 200 samples must be close.
    assert abs(er1 - er2) < 0.02
    assert abs(ce1 - ce2) < 0.05
