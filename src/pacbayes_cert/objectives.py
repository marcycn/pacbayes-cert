"""PAC-Bayes training objectives (fquad / flamb / fclassic / bbb).

Ported unchanged from PBB (the objective formulae are correct). The key
*reporting* correction is in :mod:`certificates`: the final certificate is always
computed by the same PAC-Bayes-kl inversion regardless of training objective; the
objective only shapes the posterior ``Q`` that is then certified.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class PBObjective:
    """Training-objective + empirical-risk helper bound to one posterior network."""

    def __init__(self, objective="fquad", pmin=1e-5, classes=10, delta=0.025,
                 kl_penalty=1.0, device="cuda", n_posterior=30000, n_bound=30000):
        self.objective = objective
        self.pmin = pmin
        self.classes = classes
        self.delta = delta  # confidence used *inside the training objective* (delta_train)
        self.kl_penalty = kl_penalty
        self.device = device
        self.n_posterior = n_posterior
        self.n_bound = n_bound

    def compute_empirical_risk(self, outputs, targets, bounded=True):
        risk = F.nll_loss(outputs, targets)
        if bounded:
            risk = (1.0 / (np.log(1.0 / self.pmin))) * risk
        return risk

    def compute_losses(self, net, data, target, clamping=True):
        outputs = net(data, sample=True, clamping=clamping, pmin=self.pmin)
        loss_ce = self.compute_empirical_risk(outputs, target, clamping)
        pred = outputs.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum().item()
        total = target.size(0)
        loss_01 = 1 - (correct / total)
        return loss_ce, loss_01, outputs

    def bound(self, empirical_risk, kl, n, lambda_var=None):
        """Training objective value (n = n_posterior)."""
        if self.objective == "fquad":
            kl = kl * self.kl_penalty
            ratio = torch.div(kl + np.log((2 * np.sqrt(n)) / self.delta), 2 * n)
            first = torch.sqrt(empirical_risk + ratio)
            second = torch.sqrt(ratio)
            return torch.pow(first + second, 2)
        if self.objective == "flamb":
            kl = kl * self.kl_penalty
            lamb = lambda_var.lamb_scaled
            kl_term = torch.div(kl + np.log((2 * np.sqrt(n)) / self.delta), n * lamb * (1 - lamb / 2))
            first = torch.div(empirical_risk, 1 - lamb / 2)
            return first + kl_term
        if self.objective == "fclassic":
            kl = kl * self.kl_penalty
            ratio = torch.div(kl + np.log((2 * np.sqrt(n)) / self.delta), 2 * n)
            return empirical_risk + torch.sqrt(ratio)
        if self.objective == "bbb":
            return empirical_risk + self.kl_penalty * (kl / n)
        raise ValueError(f"Wrong objective {self.objective}")

    def train_obj(self, net, data, target, clamping=True, lambda_var=None):
        kl = net.compute_kl()
        loss_ce, loss_01, outputs = self.compute_losses(net, data, target, clamping)
        obj = self.bound(loss_ce, kl, self.n_posterior, lambda_var)
        return obj, kl / self.n_posterior, outputs, loss_ce, loss_01
