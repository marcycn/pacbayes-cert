"""P0-1 / Definition-of-Done: split sample counts are exact."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pacbayes_cert.splits import make_split


def _labels(n, classes=10):
    return np.tile(np.arange(classes), n // classes + 1)[:n]


def test_learnt_full_data_60000():
    labels = _labels(60000)
    s = make_split(60000, "learnt", perc_train=1.0, perc_prior=0.5, seed_split=123, labels=labels)
    assert s.n_posterior == 60000
    assert s.n_bound == 30000
    assert s.n_prior == 30000


def test_learnt_10pct_gives_6000_3000():
    labels = _labels(60000)
    s = make_split(60000, "learnt", perc_train=0.1, perc_prior=0.5, seed_split=7, labels=labels)
    assert s.n_selected == 6000
    assert s.n_posterior == 6000
    assert s.n_bound == 3000
    assert s.n_prior == 3000


def test_perc_train_variants():
    labels = _labels(60000)
    for pt, exp_sel in [(1.0, 60000), (0.5, 30000), (0.2, 12000), (0.1, 6000)]:
        s = make_split(60000, "learnt", perc_train=pt, perc_prior=0.5, seed_split=1, labels=labels)
        assert s.n_selected == exp_sel
        assert s.n_bound == exp_sel // 2


def test_random_prior_bound_equals_selected():
    labels = _labels(60000)
    s = make_split(60000, "rand", perc_train=1.0, perc_prior=0.5, seed_split=1, labels=labels)
    assert s.n_prior == 0
    assert s.n_bound == s.n_selected == 60000
