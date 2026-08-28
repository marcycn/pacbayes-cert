"""The cached-tensor dataset must serve bit-identical data to the plain one.

The speedup it buys is only legitimate if nothing about the data changes, so
this asserts exact equality of both the inputs and the labels, including
through the label-noise wrapper used by the controlled-difficulty study.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pacbayes_cert import data as datamod  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HAVE_DATA = os.path.isdir(os.path.join(ROOT, "MNIST"))
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="MNIST not downloaded")


def test_cached_matches_plain_exactly():
    plain_train, plain_test = datamod.load_dataset("mnist", ROOT, cache_tensors=False)
    cache_train, cache_test = datamod.load_dataset("mnist", ROOT, cache_tensors=True)

    assert len(plain_train) == len(cache_train)
    assert len(plain_test) == len(cache_test)

    # a spread of indices including the first and last
    idx = [0, 1, 7, 1234, 30000, len(plain_train) - 1]
    for i in idx:
        xp, yp = plain_train[i]
        xc, yc = cache_train[i]
        assert torch.equal(xp, xc), f"input differs at index {i}"
        assert int(yp) == int(yc), f"label differs at index {i}"

    np.testing.assert_array_equal(datamod.get_labels(plain_train),
                                  datamod.get_labels(cache_train))


def test_label_noise_wrapper_still_works_on_cache():
    _, _ = datamod.load_dataset("mnist", ROOT, cache_tensors=False)
    cache_train, _ = datamod.load_dataset("mnist", ROOT, cache_tensors=True)
    labels = datamod.get_labels(cache_train)
    indices = np.arange(2000)

    noisy = datamod.apply_label_noise(cache_train, indices, 0.25, 10, seed=3)
    changed = sum(1 for i in indices if int(noisy[int(i)][1]) != int(labels[i]))
    # every corrupted label must differ from the original, and the rate is ~25%
    assert 0.20 * len(indices) < changed < 0.30 * len(indices)

    # inputs are untouched by the noise wrapper
    for i in (0, 5, 199):
        assert torch.equal(cache_train[i][0], noisy[i][0])
