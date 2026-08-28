"""P0-2 / DoD: prior and bound subsets are disjoint and cover S; stratification works."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pacbayes_cert.splits import make_split, split_hash


def _labels(n, classes=10):
    return np.tile(np.arange(classes), n // classes + 1)[:n]


def test_disjoint_and_cover():
    labels = _labels(20000)
    s = make_split(20000, "learnt", perc_train=1.0, perc_prior=0.5, seed_split=42, labels=labels)
    pri, bnd, sel = set(s.prior_indices), set(s.bound_indices), set(s.selected_indices)
    assert pri.isdisjoint(bnd)
    assert pri | bnd == sel
    s.validate()  # must not raise


def test_split_does_not_depend_on_labels():
    """The prior must be independent of the bound set, which fails if the split
    is chosen from the labels: S0 would then be picked using the class counts of
    a set containing S_b, and disjoint indices would no longer give statistical
    independence.  Permuting the labels must leave the partition untouched."""
    n = 20000
    labels = _labels(n, classes=10)
    permuted = np.random.default_rng(0).permutation(labels)
    assert not np.array_equal(labels, permuted)

    for perc_train in (1.0, 0.1):
        a = make_split(n, "learnt", perc_train, 0.5, seed_split=3, labels=labels)
        b = make_split(n, "learnt", perc_train, 0.5, seed_split=3, labels=permuted)
        assert split_hash(a) == split_hash(b), (
            f"partition changed with the labels at perc_train={perc_train}; "
            "the split is not label-independent")
        assert np.array_equal(a.prior_indices, b.prior_indices)
        assert np.array_equal(a.bound_indices, b.bound_indices)


def test_split_has_no_prefix_bias():
    """The original code took a prefix of the dataset, which correlates with any
    ordering in it.  A uniform random subset must instead spread across the whole
    index range."""
    n = 20000
    s = make_split(n, "learnt", perc_train=0.1, perc_prior=0.5, seed_split=3,
                   labels=_labels(n, classes=10))
    sel = np.asarray(s.selected_indices)
    assert sel.max() > 0.95 * n, "selection looks like a prefix"
    assert sel.min() < 0.05 * n
    # mean index of a uniform subset sits near the middle of the range
    assert abs(sel.mean() - n / 2) < 0.05 * n


def test_seed_changes_partition():
    labels = _labels(20000)
    a = make_split(20000, "learnt", 1.0, 0.5, seed_split=1, labels=labels)
    b = make_split(20000, "learnt", 1.0, 0.5, seed_split=2, labels=labels)
    assert split_hash(a) != split_hash(b)


def test_same_seed_reproduces():
    labels = _labels(20000)
    a = make_split(20000, "learnt", 1.0, 0.5, seed_split=5, labels=labels)
    b = make_split(20000, "learnt", 1.0, 0.5, seed_split=5, labels=labels)
    assert split_hash(a) == split_hash(b)


def test_stratified_is_refused_for_a_learnt_prior():
    """The rejected approach must fail loudly rather than be silently ignored.

    An ignored flag leaves the dissertation's central claim -- that no reported
    run used a label-dependent split -- resting on nobody having passed it.
    """
    import pytest

    labels = np.array([i % 10 for i in range(200)])
    with pytest.raises(ValueError, match="stratified"):
        make_split(n_total=200, prior_type="learnt", perc_train=1.0, perc_prior=0.5,
                   seed_split=0, labels=labels, stratified=True)

    # inert, and therefore allowed, where there is no prior/bound split for it to
    # have applied to: a data-free prior carves no S0
    sp = make_split(n_total=200, prior_type="rand", perc_train=1.0, perc_prior=0.5,
                    seed_split=0, labels=labels, stratified=True)
    assert sp.n_prior == 0 and sp.stratified is False
