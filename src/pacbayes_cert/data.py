"""Dataset loading and SplitInfo-driven DataLoader construction.

Loaders are built from the *explicit indices* in a :class:`~pacbayes_cert.splits.SplitInfo`
rather than from opaque samplers, so the sample sizes used in the certificate
(``n_posterior`` / ``n_bound``) provably match the data actually iterated.

Supports an optional controlled label-noise corruption (``label_noise``) applied
*only* to the bound/posterior labels of the training set, used for the controlled
difficulty experiment (addresses P1-3: a same-architecture difficulty knob).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

_NORM = {
    "mnist": ((0.1307,), (0.3081,)),
    "fashion-mnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}


class _TensorCached(torch.utils.data.Dataset):
    """A dataset whose transform has already been applied to every example.

    The preprocessing here is deterministic and identical for every example
    (``ToTensor`` then a fixed per-channel ``Normalize``; there is no
    augmentation anywhere in this project), so applying it once up front and
    serving tensors is numerically identical to applying it per ``__getitem__``.
    It is also far cheaper: profiling showed the per-item PIL decode accounted
    for about 97% of an epoch's wall-clock on this hardware, with the GPU step
    taking 0.32 s of an 11 s epoch.

    ``targets`` is kept so :func:`get_labels` keeps working unchanged.
    """

    def __init__(self, base):
        xs = [base[i][0] for i in range(len(base))]
        self.data = torch.stack(xs)
        self.targets = torch.as_tensor(get_labels(base))

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], int(self.targets[idx])


def load_dataset(name: str, root: str = "data", cache_tensors: bool = True):
    """Return ``(train, test)`` datasets with the standard normalisation applied.

    With ``cache_tensors`` the transform is applied once to the whole dataset and
    the result is served from memory; see :class:`_TensorCached` for why this is
    numerically equivalent.  Pass ``cache_tensors=False`` to get the plain
    torchvision datasets, which is what the equivalence test compares against.
    """
    if name not in _NORM:
        raise ValueError(f"unknown dataset {name!r}")
    mean, std = _NORM[name]
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if name == "mnist":
        train = datasets.MNIST(root, train=True, download=True, transform=tfm)
        test = datasets.MNIST(root, train=False, download=True, transform=tfm)
    elif name == "fashion-mnist":
        train = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
        test = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)
    else:
        train = datasets.CIFAR10(root, train=True, download=True, transform=tfm)
        test = datasets.CIFAR10(root, train=False, download=True, transform=tfm)
    if cache_tensors:
        train, test = _TensorCached(train), _TensorCached(test)
    return train, test


def get_labels(dataset) -> np.ndarray:
    """Extract integer labels as a numpy array (works for MNIST/FMNIST/CIFAR)."""
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [dataset[i][1] for i in range(len(dataset))]
    if isinstance(targets, torch.Tensor):
        return targets.cpu().numpy()
    return np.asarray(targets)


class _LabelNoiseDataset(torch.utils.data.Dataset):
    """Wrap a dataset and flip a fixed fraction of labels (deterministic by seed)."""

    def __init__(self, base, noisy_targets: dict):
        self.base = base
        self.noisy = noisy_targets  # index -> corrupted label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if idx in self.noisy:
            y = self.noisy[idx]
        return x, y


def apply_label_noise(
    train, indices: np.ndarray, noise: float, n_classes: int, seed: int
) -> "_LabelNoiseDataset":
    """Corrupt ``noise`` fraction of the labels of ``indices`` to a uniformly random
    *different* class.  Used only for controlled-difficulty experiments."""
    rng = np.random.default_rng(seed)
    labels = get_labels(train)
    flip = rng.random(len(indices)) < noise
    noisy = {}
    for idx, do in zip(indices, flip):
        if do:
            orig = int(labels[idx])
            new = int(rng.integers(0, n_classes - 1))
            if new >= orig:
                new += 1
            noisy[int(idx)] = new
    return _LabelNoiseDataset(train, noisy)


def build_loaders(
    train,
    test,
    split,
    batch_size: int,
    loader_generator: torch.Generator,
    test_batch_size: Optional[int] = None,
    loader_kwargs: Optional[dict] = None,
    mc_eval_batch: int = 0,
) -> dict:
    """Build all loaders required by the pipeline from an explicit SplitInfo.

    Returns a dict with:
      * ``posterior``   : shuffled loader over ``S`` (posterior training).
      * ``prior``       : shuffled loader over ``S0`` (learn prior; None if rand).
      * ``bound``       : loader over ``S \\ S0`` in mini-batches (certificate eval).
      * ``bound_1batch``: same data in a single batch (efficiency for small sets).
      * ``test``        : shuffled test loader (mini-batch).
      * ``test_1batch`` : single-batch test loader.
    """
    loader_kwargs = loader_kwargs or {}
    test_batch_size = test_batch_size or batch_size

    sel = list(split.selected_indices)
    bnd = list(split.bound_indices)
    pri = list(split.prior_indices)

    post_set = Subset(train, sel)
    bound_set = Subset(train, bnd)

    posterior = DataLoader(
        post_set, batch_size=batch_size, shuffle=True, generator=loader_generator, **loader_kwargs
    )
    bound = DataLoader(bound_set, batch_size=batch_size, shuffle=False, **loader_kwargs)
    bound_1batch = DataLoader(bound_set, batch_size=len(bnd), shuffle=False, **loader_kwargs)
    # Large-batch loader for the Monte-Carlo certificate: drastically fewer Python
    # iterations than batch_size=250. 0 -> whole bound set in one batch (FCN);
    # a positive value caps the batch (deep CNNs, to bound GPU memory).
    mc_bs = len(bnd) if mc_eval_batch <= 0 else min(mc_eval_batch, len(bnd))
    bound_mc = DataLoader(bound_set, batch_size=mc_bs, shuffle=False, **loader_kwargs)

    prior = None
    if split.prior_type == "learnt" and pri:
        prior_set = Subset(train, pri)
        prior = DataLoader(
            prior_set, batch_size=batch_size, shuffle=True, generator=loader_generator, **loader_kwargs
        )

    test_loader = DataLoader(test, batch_size=test_batch_size, shuffle=True, **loader_kwargs)
    test_1batch = DataLoader(test, batch_size=len(test), shuffle=False, **loader_kwargs)

    return {
        "posterior": posterior,
        "prior": prior,
        "bound": bound,
        "bound_1batch": bound_1batch,
        "bound_mc": bound_mc,
        "test": test_loader,
        "test_1batch": test_1batch,
    }
