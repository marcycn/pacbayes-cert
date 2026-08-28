"""Explicit, decomposed random-seed control (fixes P0-7).

The original PBB pipeline silently reset the PyTorch / NumPy global seeds to the
hard-coded constants 7, 0 and 10 inside ``utils.runexp`` and ``data.loadbatches``.
That made the command-line ``--seed`` pure metadata: every "seed" produced an
identical run, so no genuine multi-seed variance could be measured.

Here randomness is split into four *named*, independently-passed streams, whose
values all originate in this module:

* ``seed_split``   - which examples land in the prior / bound / posterior subset.
* ``seed_model``   - parameter initialisation and posterior weight sampling during training.
* ``seed_loader``  - DataLoader shuffling order.
* ``seed_mc``      - posterior weight sampling for the final Monte-Carlo certificate.
* ``seed_eval``    - posterior weight sampling for the descriptive test-set metrics.

``seed_eval`` exists because the three deployment metrics run after the
certificate and also sample posterior weights.  Left to continue from wherever
the certificate's ``m`` draws finished, they would depend on ``mc_samples`` and
on how the bound set was batched -- so raising the Monte-Carlo budget would move
a *test error* that has nothing to do with it.  Reseeding here decouples them.

A :class:`SeedBundle` is derived deterministically from a single integer
``--seed`` so that one CLI number still reproduces a whole run, while different
numbers genuinely change every stream.

How a stream reaches the code that consumes it differs by stream, and the
difference is worth stating so the docstring is not read as stronger than it is.
``seed_split`` and ``seed_loader`` are threaded through as explicit generator
objects.  ``seed_model`` and ``seed_mc`` are applied by reseeding the global
Torch / NumPy RNGs at the two points where they take effect, because the layers
that sample posterior weights draw from the global generator.  So this module is
where every seed *value* comes from, but it is not the only place that calls
``torch.manual_seed``; :mod:`pacbayes_cert.runner` does so for ``seed_mc``
immediately before the certificate is computed.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np


def _derive(base_seed: int, name: str) -> int:
    """Derive a stable 32-bit sub-seed from ``(base_seed, name)``.

    Uses BLAKE2b so the mapping is deterministic across processes and machines
    (Python's built-in ``hash`` is salted per-process and must not be used).
    """
    h = hashlib.blake2b(f"{base_seed}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % (2**32)


@dataclass(frozen=True)
class SeedBundle:
    """The four independent seed streams for one run."""

    base: int
    seed_split: int
    seed_model: int
    seed_loader: int
    seed_mc: int
    seed_eval: int

    @classmethod
    def from_base(cls, base_seed: int) -> "SeedBundle":
        return cls(
            base=base_seed,
            seed_split=_derive(base_seed, "split"),
            seed_model=_derive(base_seed, "model"),
            seed_loader=_derive(base_seed, "loader"),
            seed_mc=_derive(base_seed, "mc"),
            seed_eval=_derive(base_seed, "eval"),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def seed_model_rngs(seed_model: int) -> None:
    """Seed the *global* Python/NumPy/PyTorch RNGs used for model init & training.

    Call this exactly once, immediately before constructing and training the
    networks.  Data splitting and DataLoader ordering must use their own
    generators (``rng_for_split`` / ``generator_for_loader``) so that changing,
    say, the model seed does not perturb the data partition.
    """
    import random

    import torch

    random.seed(seed_model)
    np.random.seed(seed_model)
    torch.manual_seed(seed_model)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_model)


def set_deterministic(deterministic: bool = True) -> None:
    """Configure cuDNN for reproducibility (mirrors the original intent)."""
    import torch

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def rng_for_split(seed_split: int) -> np.random.Generator:
    """Independent NumPy Generator for the data partition (never the global RNG)."""
    return np.random.default_rng(seed_split)


def generator_for_loader(seed_loader: int):
    """Explicit torch.Generator so DataLoader shuffling is reproducible & controlled."""
    import torch

    g = torch.Generator()
    g.manual_seed(seed_loader)
    return g


def generator_for_mc(seed_mc: int, device: str = "cpu"):
    """Explicit torch.Generator for final Monte-Carlo posterior weight sampling."""
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed(seed_mc)
    return g
