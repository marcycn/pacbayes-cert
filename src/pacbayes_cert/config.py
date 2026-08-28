"""Experiment configuration: dataclass + YAML loading with defaults."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class ExpConfig:
    dataset: str = "mnist"            # mnist | fashion-mnist | cifar10
    model: str = "fcn"               # fcn | cnn | cnn13 | cnn15 (the last two are CIFAR-10 only)
    objective: str = "fquad"         # fquad | flamb | fclassic | bbb
    prior_type: str = "learnt"       # rand | learnt
    base_seed: int = 0

    perc_train: float = 1.0
    perc_prior: float = 0.5
    stratified: bool = False   # retained for older configs; the split never uses labels
    label_noise: float = 0.0

    sigma_prior: float = 0.03
    prior_dist: str = "gaussian"
    pmin: float = 1e-5
    kl_penalty: float = 0.1
    initial_lamb: float = 6.0
    dropout_prob: float = 0.2

    lr: float = 0.005
    momentum: float = 0.95
    lr_prior: float = 0.005
    momentum_prior: float = 0.99
    train_epochs: int = 100
    prior_epochs: int = 70
    batch_size: int = 250

    # confidence budget (split so joint confidence is honest, P0-5)
    delta_train: float = 0.025
    delta_mc: float = 0.005
    delta_pac: float = 0.005
    mc_samples: int = 10000
    samples_ensemble: int = 100
    # batch size for the MC certificate / ensemble eval. 0 = use the whole bound
    # set in a single batch (best for FCN; cuts Python/kernel-launch overhead ~100x).
    # Set to a moderate value (e.g. 2500) for deep CNNs to bound GPU memory.
    mc_eval_batch: int = 0

    device: str = "cuda"

    def __post_init__(self):
        # PyYAML parses unsuffixed scientific notation like ``1e-5`` as a *string*
        # (no decimal point), so coerce all numeric fields defensively.
        int_fields = {"base_seed", "train_epochs", "prior_epochs", "batch_size",
                      "mc_samples", "samples_ensemble", "mc_eval_batch"}
        float_fields = {"perc_train", "perc_prior", "label_noise", "sigma_prior",
                        "pmin", "kl_penalty", "initial_lamb", "dropout_prob", "lr",
                        "momentum", "lr_prior", "momentum_prior", "delta_train",
                        "delta_mc", "delta_pac"}
        for f in int_fields:
            object.__setattr__(self, f, int(getattr(self, f)))
        for f in float_fields:
            object.__setattr__(self, f, float(getattr(self, f)))
        object.__setattr__(self, "stratified", bool(self.stratified))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_yaml(path) -> "ExpConfig":
        import yaml

        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return ExpConfig(**d)

    def merged(self, **overrides) -> "ExpConfig":
        d = self.to_dict()
        d.update({k: v for k, v in overrides.items() if v is not None})
        return ExpConfig(**d)
