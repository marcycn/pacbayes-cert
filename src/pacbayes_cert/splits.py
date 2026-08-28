"""Explicit, immutable data partitioning (fixes P0-1, P0-2, P1-9).

Original problems
-----------------
* ``utils.py`` set ``n_bound = len(val_bound.dataset)`` which returns the size of
  the *underlying* dataset (e.g. 60 000), not the bound subset actually selected
  by the sampler.  This systematically shrank the complexity term ``KL/n`` and the
  confidence term, producing certificates that were too tight (P0-1).
* The paper described the prior subset as disjoint from *both* the posterior
  training set and the bound set, but the code trains the posterior on the whole
  selected set ``S`` (including the prior subset) and only the prior subset is
  disjoint from the bound subset (P0-2).
* For ``perc_train < 1`` the original took ``range(new_num_train)`` -- a *prefix*
  of the dataset -- before shuffling, so small-data subsets were not a random
  sample of the full training set and could carry ordering / class bias (P1-9).
  Replacing that prefix with *class-stratified* sampling removes the bias but
  makes the partition depend on the labels, which breaks the independence the
  PAC-Bayes prior condition needs; a uniform random permutation is used instead.

This module replaces all of that with an explicit :class:`SplitInfo` carrying the
exact integer indices for every subset, a label-independent partition, and runtime
assertions on disjointness and sample counts.

Definitions (consistent with the corrected paper)
-------------------------------------------------
* ``S``      = ``selected_indices``  : the ``perc_train`` fraction kept for the experiment.
* ``S0``     = ``prior_indices``     : ``perc_prior`` of ``S``, used to learn the data-dependent prior.
* ``S \\ S0`` = ``bound_indices``     : the disjoint remainder, used for the empirical risk in the certificate.

``n_posterior = |S|`` (the posterior may use all of ``S``), ``n_bound = |S \\ S0|``.
For a data-free (random) prior there is no ``S0``; then ``bound_indices == S`` and
``n_bound = |S|``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SplitInfo:
    """Immutable record of one data partition.

    All index tuples are stored sorted for stable hashing; sampling order is
    controlled separately by the DataLoader generator.
    """

    n_total: int
    n_selected: int
    n_prior: int
    n_posterior: int
    n_bound: int
    prior_type: str  # "rand" | "learnt"
    perc_train: float
    perc_prior: float
    stratified: bool
    seed_split: int
    selected_indices: Tuple[int, ...]
    prior_indices: Tuple[int, ...]
    bound_indices: Tuple[int, ...]
    prior_class_hist: Tuple[int, ...] = field(default=())
    bound_class_hist: Tuple[int, ...] = field(default=())

    def validate(self) -> None:
        """Runtime assertions enforcing the contract (run before every experiment)."""
        sel, pri, bnd = set(self.selected_indices), set(self.prior_indices), set(self.bound_indices)
        assert pri.isdisjoint(bnd), "prior and bound subsets must be disjoint"
        assert pri | bnd == sel, "prior ∪ bound must equal the selected set S"
        assert self.n_posterior == self.n_selected == len(sel), "n_posterior must equal |S|"
        assert self.n_bound == len(bnd) == len(self.bound_indices), "n_bound must equal |bound|"
        assert self.n_prior == len(pri), "n_prior must equal |prior|"
        if self.prior_type == "rand":
            assert self.n_prior == 0 and bnd == sel, "random prior: no prior subset, bound == S"

    def summary(self) -> Dict[str, int]:
        return {
            "n_total": self.n_total,
            "n_selected": self.n_selected,
            "n_prior": self.n_prior,
            "n_posterior": self.n_posterior,
            "n_bound": self.n_bound,
        }


def make_split(
    n_total: int,
    prior_type: str,
    perc_train: float,
    perc_prior: float,
    seed_split: int,
    labels: Optional[np.ndarray] = None,
    stratified: bool = False,
) -> SplitInfo:
    """Construct an immutable, validated :class:`SplitInfo`.

    Parameters
    ----------
    n_total : int
        Size of the full training dataset.
    prior_type : {"rand", "learnt"}
        ``"learnt"`` carves a prior subset ``S0`` out of ``S``; ``"rand"`` does not.
    perc_train : float
        Fraction of the full set kept as the selected set ``S``.
    perc_prior : float
        Fraction of ``S`` used to learn the prior (only for ``prior_type='learnt'``).
    seed_split : int
        Seed for the dedicated split RNG (independent of model/loader seeds).
    labels : np.ndarray, optional
        Per-example class labels.  Used only to record the class histograms of the
        resulting subsets; the partition itself never reads them.
    stratified : bool
        Must be falsy for a learnt prior.  Accepted (and inert) otherwise, since
        older stored configurations carry ``stratified: true`` on data-free-prior
        cells where there is no prior/bound split for it to have applied to.

    Raises
    ------
    ValueError
        If a class-stratified partition is requested for a learnt prior.  Silently
        ignoring the flag would leave the paper's central claim -- that no
        reported run used a label-dependent split -- resting on nobody having
        passed it, which is not an enforcement.
    """
    if stratified and prior_type == "learnt":
        raise ValueError(
            "stratified=True is not supported for a learnt prior: drawing S0 "
            "against class counts taken over the whole of S makes the prior "
            "depend on the bound set's labels, which breaks the PAC-Bayes prior "
            "condition. Set stratified=false in the configuration."
        )

    rng = np.random.default_rng(seed_split)
    all_idx = np.arange(n_total)

    # Both the selection of S and its split into S0 and S_b are drawn from the
    # split RNG alone and never look at the labels.  This is a validity
    # requirement, not a stylistic choice.  Class-stratified sampling was
    # introduced here to remove the prefix bias of the original code, and it does
    # remove it, but it also decides which examples enter S0 from the class counts
    # of the whole of S, and S contains S_b.  The prior would then depend on the
    # labels of the bound set, so disjoint index sets would no longer deliver the
    # statistical independence the PAC-Bayes prior condition needs; and a sample
    # with fixed class quotas is not an i.i.d. draw from D, which is what the
    # concentration result behind the bound assumes.  A uniform random
    # permutation removes the prefix bias just as effectively and preserves both
    # properties, so it is what we use.
    #
    # To be exact about what "label-independent" means here: no label is consulted
    # in deciding membership of S, S0 or S_b.  Labels are read once further down,
    # after the partition is fixed, only to record the class histograms of the
    # resulting subsets; nothing derived from them feeds back into membership,
    # n_bound, or the choice of prior training data.
    n_selected = int(round(perc_train * n_total))
    perm = all_idx.copy()
    rng.shuffle(perm)
    selected = np.sort(perm[:n_selected])

    if prior_type == "learnt":
        n_prior = int(round(perc_prior * n_selected))
        sel_perm = selected.copy()
        rng.shuffle(sel_perm)
        prior = sel_perm[:n_prior]
        prior_set = set(int(i) for i in prior)
        bound = np.array([i for i in selected if int(i) not in prior_set], dtype=int)
    else:  # random / data-free prior: no S0, bound == S
        n_prior = 0
        prior = np.array([], dtype=int)
        bound = selected.copy()

    prior = np.sort(prior)
    bound = np.sort(bound)

    prior_hist: Tuple[int, ...] = ()
    bound_hist: Tuple[int, ...] = ()
    if labels is not None:
        n_classes = int(labels.max()) + 1
        if len(prior):
            prior_hist = tuple(int(c) for c in np.bincount(labels[prior], minlength=n_classes))
        bound_hist = tuple(int(c) for c in np.bincount(labels[bound], minlength=n_classes))

    info = SplitInfo(
        n_total=n_total,
        n_selected=int(n_selected),
        n_prior=int(n_prior),
        n_posterior=int(n_selected),
        n_bound=int(len(bound)),
        prior_type=prior_type,
        perc_train=perc_train,
        perc_prior=perc_prior,
        stratified=False,   # the partition is label-independent by construction
        seed_split=seed_split,
        selected_indices=tuple(int(i) for i in selected),
        prior_indices=tuple(int(i) for i in prior),
        bound_indices=tuple(int(i) for i in bound),
        prior_class_hist=prior_hist,
        bound_class_hist=bound_hist,
    )
    info.validate()
    return info


def split_hash(info: SplitInfo) -> str:
    """Stable content hash of the selected/prior/bound indices for provenance."""
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    for name in ("selected_indices", "prior_indices", "bound_indices"):
        arr = np.asarray(getattr(info, name), dtype=np.int64)
        h.update(name.encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def assert_loaders_match_split(loaders: dict, split: SplitInfo) -> None:
    """Check at run time that the loaders really carry the partition that was saved.

    ``verify_splits.py`` establishes that the indices written to
    ``split_indices.npz`` are the ones the label-independent generator produces.
    That is a statement about how the indices were *made*, not about what the
    training and evaluation actually consumed: in principle a second sampler
    could have fed the certificate a different subset, and nothing downstream
    would notice.

    This closes that gap from the other end, by reading the index list off each
    ``Subset`` the loaders were built on and comparing it with the partition
    recorded alongside the run. It costs one list comparison per experiment and
    turns "the same SplitInfo was passed to both" from a property of the source
    into a property of the run.
    """
    def indices_of(loader):
        ds = loader.dataset
        return list(getattr(ds, "indices", []))

    def assert_consumes_all(loader, name):
        """Check the sampler covers its dataset exactly once.

        Matching ``Subset.indices`` establishes the loader's *support*, not what
        it draws from that support: a sampler that sub-sampled, dropped a
        partial batch, or drew with replacement would leave the support right
        and the estimate wrong.  For the bound loaders the sampler is
        sequential, so its output can be compared directly and cheaply, and it
        carries no RNG to disturb.
        """
        if getattr(loader, "drop_last", False):
            raise AssertionError(
                f"loader {name!r} drops its last partial batch, so the "
                f"Monte-Carlo denominator would count examples it never scored"
            )
        n = len(loader.dataset)
        order = list(loader.sampler)
        if sorted(order) != list(range(n)):
            raise AssertionError(
                f"loader {name!r} does not draw each of its {n} examples exactly "
                f"once ({len(order)} draws, {len(set(order))} distinct)"
            )

    for name in ("bound", "bound_1batch", "bound_mc"):
        got = indices_of(loaders[name])
        if got != list(split.bound_indices):
            raise AssertionError(
                f"loader {name!r} does not carry S_b: {len(got)} indices against "
                f"{split.n_bound} recorded; the certificate would be evaluated on "
                f"a different subset from the one saved with the run"
            )
        assert_consumes_all(loaders[name], name)

    if indices_of(loaders["posterior"]) != list(split.selected_indices):
        raise AssertionError("posterior loader does not carry S")
    # The posterior loader shuffles from a seeded generator, so its sampler is
    # not iterated here: doing so would advance that generator and change the
    # training order the run is supposed to reproduce.  Only the properties
    # readable without drawing are checked.
    if getattr(loaders["posterior"], "drop_last", False):
        raise AssertionError("posterior loader drops its last partial batch")
    if len(loaders["posterior"].sampler) != split.n_posterior:
        raise AssertionError("posterior sampler does not span all of S")

    if split.prior_type == "learnt":
        if indices_of(loaders["prior"]) != list(split.prior_indices):
            raise AssertionError("prior loader does not carry S0")
    elif loaders.get("prior") is not None:
        raise AssertionError("a data-free prior must not have a prior loader")
