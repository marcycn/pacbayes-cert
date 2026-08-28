"""Two properties the certificate silently depends on, made explicit.

1. The loader the Monte-Carlo estimate runs over must contain exactly S_b, and
   nothing from S_0.  If a single S_0 example leaked into it the prior would no
   longer be independent of the set the empirical risk is measured on, and
   nothing else in the pipeline would notice.
2. The bounded cross-entropy surrogate must actually land in [0, 1].  PAC-Bayes
   needs a [0, 1]-valued loss; the clamp-and-rescale is what buys that, and an
   off-by-one in the rescaling constant would break the guarantee while leaving
   every number looking plausible.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pacbayes_cert.data import build_loaders        # noqa: E402
from pacbayes_cert.models import output_transform   # noqa: E402
from pacbayes_cert.objectives import PBObjective    # noqa: E402
from pacbayes_cert.splits import make_split         # noqa: E402


class _Toy(torch.utils.data.Dataset):
    """Returns its own index as the input, so a loader's contents are readable."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.tensor([float(i)]), int(i) % 10


def _indices_seen(loader):
    seen = []
    for x, _ in loader:
        seen.extend(int(v) for v in x.view(-1).tolist())
    return sorted(seen)


def test_mc_loader_is_exactly_the_bound_set():
    n = 400
    ds = _Toy(n)
    labels = np.array([i % 10 for i in range(n)])
    split = make_split(n_total=n, prior_type="learnt", perc_train=1.0,
                       perc_prior=0.5, seed_split=11, labels=labels)
    loaders = build_loaders(ds, ds, split, batch_size=32,
                            loader_generator=torch.Generator().manual_seed(0))

    bnd = sorted(int(i) for i in split.bound_indices)
    pri = set(int(i) for i in split.prior_indices)

    for name in ("bound", "bound_1batch", "bound_mc"):
        seen = _indices_seen(loaders[name])
        assert seen == bnd, name
        assert not (set(seen) & pri), "%s leaks S0 examples" % name

    # and the posterior loader is allowed all of S, which is the asymmetry the
    # theory turns on
    assert sorted(set(_indices_seen(loaders["posterior"]))) == sorted(
        int(i) for i in split.selected_indices)


def test_mc_loader_covers_the_bound_set_in_one_pass_when_batched():
    n = 500
    ds = _Toy(n)
    labels = np.array([i % 10 for i in range(n)])
    split = make_split(n_total=n, prior_type="learnt", perc_train=1.0,
                       perc_prior=0.5, seed_split=3, labels=labels)
    # mc_eval_batch smaller than |S_b| forces several batches; every example must
    # still appear exactly once, since the MC denominator counts examples
    loaders = build_loaders(ds, ds, split, batch_size=32,
                            loader_generator=torch.Generator().manual_seed(0),
                            mc_eval_batch=37)
    seen = _indices_seen(loaders["bound_mc"])
    assert seen == sorted(int(i) for i in split.bound_indices)
    assert len(seen) == len(set(seen)) == split.n_bound


def test_bounded_cross_entropy_is_in_unit_interval():
    """The clamp at log(pmin) plus the 1/log(1/pmin) rescaling is what makes the
    surrogate [0, 1]-valued.

    This drives the *production* path -- ``output_transform`` from models.py and
    ``PBObjective.compute_empirical_risk`` -- rather than a formula rewritten in
    the test, so that dropping the clamp in the source makes it fail.
    """
    pmin = 1e-5
    classes = 10
    pb = PBObjective(objective="fquad", pmin=pmin, classes=classes, delta=0.025,
                     kl_penalty=0.1, device="cpu", n_posterior=100, n_bound=50)
    target = torch.arange(classes)

    def risk(logits):
        return float(pb.compute_empirical_risk(
            output_transform(logits, clamping=True, pmin=pmin), target, bounded=True))

    # a perfectly confident correct predictor: the bottom of the range
    best = torch.full((classes, classes), -30.0)
    best[torch.arange(classes), target] = 30.0
    assert 0.0 <= risk(best) < 1e-6

    # a perfectly confident *wrong* one: without the clamp the log-probability
    # diverges, so this is the case that pins the top of the range
    worst = torch.full((classes, classes), 30.0)
    worst[torch.arange(classes), target] = -30.0
    assert 0.999 < risk(worst) <= 1.0 + 1e-9

    # random logits at a scale that would blow past 1 unclamped
    torch.manual_seed(0)
    for _ in range(200):
        assert 0.0 <= risk(torch.randn(classes, classes) * 20.0) <= 1.0 + 1e-9


def test_unclamped_cross_entropy_would_leave_the_unit_interval():
    """The complement of the test above: with clamping off the same input does
    exceed 1, which is what makes the assertion above load-bearing rather than
    vacuously true for any input we happened to pick."""
    pmin = 1e-5
    classes = 10
    pb = PBObjective(objective="fquad", pmin=pmin, classes=classes, delta=0.025,
                     kl_penalty=0.1, device="cpu", n_posterior=100, n_bound=50)
    target = torch.arange(classes)
    worst = torch.full((classes, classes), 30.0)
    worst[torch.arange(classes), target] = -30.0
    unclamped = pb.compute_empirical_risk(
        output_transform(worst, clamping=False), target, bounded=True)
    assert float(unclamped) > 1.0


def test_runtime_assertion_catches_a_loader_that_lost_the_split():
    """The runner asserts, per experiment, that the loaders carry the partition
    it is about to save. Without that, a second sampler could feed the
    certificate a different subset and nothing downstream would notice: the
    saved indices would still verify, because they are correct -- they just would
    not be the ones used."""
    import pytest

    from pacbayes_cert.splits import assert_loaders_match_split

    n = 400
    ds = _Toy(n)
    labels = np.array([i % 10 for i in range(n)])
    split = make_split(n_total=n, prior_type="learnt", perc_train=1.0,
                       perc_prior=0.5, seed_split=11, labels=labels)
    loaders = build_loaders(ds, ds, split, batch_size=32,
                            loader_generator=torch.Generator().manual_seed(0))

    assert_loaders_match_split(loaders, split)          # the honest case passes

    # swap the MC evaluator onto the wrong subset, as a stray sampler would
    loaders["bound_mc"] = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, [int(i) for i in split.selected_indices]),
        batch_size=32, shuffle=False)
    with pytest.raises(AssertionError, match="bound_mc"):
        assert_loaders_match_split(loaders, split)


def test_runtime_assertion_catches_a_sampler_that_does_not_cover_the_bound_set():
    """Matching Subset.indices fixes the loader's support, not what it draws
    from that support. A sampler that sub-samples, or a loader that drops its
    last partial batch, leaves the support correct while scoring fewer examples
    than the Monte-Carlo denominator counts."""
    import pytest

    from pacbayes_cert.splits import assert_loaders_match_split

    n = 400
    ds = _Toy(n)
    labels = np.array([i % 10 for i in range(n)])
    split = make_split(n_total=n, prior_type="learnt", perc_train=1.0,
                       perc_prior=0.5, seed_split=11, labels=labels)
    loaders = build_loaders(ds, ds, split, batch_size=32,
                            loader_generator=torch.Generator().manual_seed(0))
    bound_set = loaders["bound_mc"].dataset

    # right support, but only half of it is ever drawn
    half = list(range(len(bound_set) // 2))
    loaders["bound_mc"] = torch.utils.data.DataLoader(
        bound_set, batch_size=32, sampler=half)
    with pytest.raises(AssertionError, match="exactly once"):
        assert_loaders_match_split(loaders, split)

    # right support, drawn in full, but the trailing partial batch is discarded
    loaders["bound_mc"] = torch.utils.data.DataLoader(
        bound_set, batch_size=32, shuffle=False, drop_last=True)
    with pytest.raises(AssertionError, match="drops its last partial batch"):
        assert_loaders_match_split(loaders, split)
