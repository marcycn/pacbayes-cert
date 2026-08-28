"""Probabilistic and deterministic network layers / architectures.

Faithfully ported from the PBB reference implementation of Perez-Ortiz et al.
(2021), "Tighter risk certificates for neural networks" (JMLR). The probabilistic
math (Gaussian/Laplace KL, sampling, truncated-normal init) is unchanged because
it is correct; only obvious latent bugs in rarely-used branches are fixed and
noted.  Provenance is recorded in ``third_party/PBB`` and ``patches/``.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(l, u)
        tensor.mul_(2)
        tensor.sub_(1)
        eps = torch.finfo(tensor.dtype).eps
        tensor.clamp_(min=-(1.0 - eps), max=(1.0 - eps))
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Gaussian(nn.Module):
    """Diagonal Gaussian with softplus std parameterisation (std = log(1+exp(rho)))."""

    def __init__(self, mu, rho, device="cuda", fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device

    @property
    def sigma(self):
        return torch.log(1 + torch.exp(self.rho))

    def sample(self):
        epsilon = torch.randn(self.sigma.size()).to(self.device)
        return self.mu + self.sigma * epsilon

    def compute_kl(self, other):
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)
        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(torch.pow(self.mu - other.mu, 2), b0)
        term3 = torch.div(b1, b0)
        return (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()


class Laplace(nn.Module):
    """Diagonal Laplace with softplus scale parameterisation."""

    def __init__(self, mu, rho, device="cuda", fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device

    @property
    def scale(self):
        m = nn.Softplus()
        return m(self.rho)

    def sample(self):
        epsilon = (0.999 * torch.rand(self.scale.size()) - 0.49999).to(self.device)
        return self.mu - torch.mul(
            torch.mul(self.scale, torch.sign(epsilon)), torch.log(1 - 2 * torch.abs(epsilon))
        )

    def compute_kl(self, other):
        b1 = self.scale
        b0 = other.scale
        term1 = torch.log(torch.div(b0, b1))
        aux = torch.abs(self.mu - other.mu)
        term2 = torch.div(aux, b0)
        term3 = torch.div(b1, b0) * torch.exp(torch.div(-aux, b1))
        return (term1 + term2 + term3 - 1).sum()


class Linear(nn.Module):
    """Deterministic linear layer with truncated-normal init (for fair comparison)."""

    def __init__(self, in_features, out_features, device="cuda"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        sigma_weights = 1 / np.sqrt(in_features)
        self.weight = nn.Parameter(
            trunc_normal_(torch.Tensor(out_features, in_features), 0, sigma_weights,
                          -2 * sigma_weights, 2 * sigma_weights),
            requires_grad=True,
        )
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class Lambda_var(nn.Module):
    """Trainable lambda for the flamb objective, scaled to (1/sqrt(n), 1)."""

    def __init__(self, lamb, n):
        super().__init__()
        self.lamb = nn.Parameter(torch.tensor([lamb]), requires_grad=True)
        self.min = 1 / np.sqrt(n)

    @property
    def lamb_scaled(self):
        m = nn.Sigmoid()
        return m(self.lamb) * (1 - self.min) + self.min


def _pick_dist(prior_dist):
    if prior_dist == "gaussian":
        return Gaussian
    if prior_dist == "laplace":
        return Laplace
    raise ValueError(f"Wrong prior_dist {prior_dist}")


class ProbLinear(nn.Module):
    def __init__(self, in_features, out_features, rho_prior, prior_dist="gaussian", device="cuda",
                 init_prior="weights", init_layer=None, init_layer_prior=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        sigma_weights = 1 / np.sqrt(in_features)

        if init_layer:
            weights_mu_init = init_layer.weight
            bias_mu_init = init_layer.bias
        else:
            weights_mu_init = trunc_normal_(torch.Tensor(out_features, in_features), 0,
                                            sigma_weights, -2 * sigma_weights, 2 * sigma_weights)
            bias_mu_init = torch.zeros(out_features)

        weights_rho_init = torch.ones(out_features, in_features) * rho_prior
        bias_rho_init = torch.ones(out_features) * rho_prior

        if init_prior == "zeros":
            weights_mu_prior = torch.zeros(out_features, in_features)
            bias_mu_prior = torch.zeros(out_features)
        elif init_prior == "random":
            weights_mu_prior = trunc_normal_(torch.Tensor(out_features, in_features), 0,
                                             sigma_weights, -2 * sigma_weights, 2 * sigma_weights)
            bias_mu_prior = torch.zeros(out_features)
        elif init_prior == "weights":
            if init_layer_prior:
                weights_mu_prior = init_layer_prior.weight
                bias_mu_prior = init_layer_prior.bias
            else:
                weights_mu_prior = weights_mu_init
                bias_mu_prior = bias_mu_init
        else:
            raise ValueError("Wrong type of prior initialisation!")

        dist = _pick_dist(prior_dist)
        self.bias = dist(bias_mu_init.clone(), bias_rho_init.clone(), device=device, fixed=False)
        self.weight = dist(weights_mu_init.clone(), weights_rho_init.clone(), device=device, fixed=False)
        self.weight_prior = dist(weights_mu_prior.clone(), weights_rho_init.clone(), device=device, fixed=True)
        self.bias_prior = dist(bias_mu_prior.clone(), bias_rho_init.clone(), device=device, fixed=True)
        self.kl_div = 0

    def forward(self, x, sample=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training:
            self.kl_div = self.weight.compute_kl(self.weight_prior) + self.bias.compute_kl(self.bias_prior)
        return F.linear(x, weight, bias)


class ProbConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rho_prior, prior_dist="gaussian",
                 device="cuda", stride=1, padding=0, dilation=1, init_prior="weights",
                 init_layer=None, init_layer_prior=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1

        in_features = self.in_channels
        for k in self.kernel_size:
            in_features *= k
        sigma_weights = 1 / np.sqrt(in_features)

        if init_layer:
            weights_mu_init = init_layer.weight
            bias_mu_init = init_layer.bias
        else:
            weights_mu_init = trunc_normal_(torch.Tensor(out_channels, in_channels, *self.kernel_size),
                                            0, sigma_weights, -2 * sigma_weights, 2 * sigma_weights)
            bias_mu_init = torch.zeros(out_channels)

        weights_rho_init = torch.ones(out_channels, in_channels, *self.kernel_size) * rho_prior
        bias_rho_init = torch.ones(out_channels) * rho_prior

        # NOTE: original PBB referenced undefined ``out_features`` in the zeros/random
        # branches; corrected here to the proper conv shapes.
        if init_prior == "zeros":
            weights_mu_prior = torch.zeros(out_channels, in_channels, *self.kernel_size)
            bias_mu_prior = torch.zeros(out_channels)
        elif init_prior == "random":
            weights_mu_prior = trunc_normal_(torch.Tensor(out_channels, in_channels, *self.kernel_size),
                                             0, sigma_weights, -2 * sigma_weights, 2 * sigma_weights)
            bias_mu_prior = torch.zeros(out_channels)
        elif init_prior == "weights":
            if init_layer_prior:
                weights_mu_prior = init_layer_prior.weight
                bias_mu_prior = init_layer_prior.bias
            else:
                weights_mu_prior = weights_mu_init
                bias_mu_prior = bias_mu_init
        else:
            raise ValueError("Wrong type of prior initialisation!")

        dist = _pick_dist(prior_dist)
        self.weight = dist(weights_mu_init.clone(), weights_rho_init.clone(), device=device, fixed=False)
        self.bias = dist(bias_mu_init.clone(), bias_rho_init.clone(), device=device, fixed=False)
        self.weight_prior = dist(weights_mu_prior.clone(), weights_rho_init.clone(), device=device, fixed=True)
        self.bias_prior = dist(bias_mu_prior.clone(), bias_rho_init.clone(), device=device, fixed=True)
        self.kl_div = 0

    def forward(self, x, sample=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training:
            self.kl_div = self.weight.compute_kl(self.weight_prior) + self.bias.compute_kl(self.bias_prior)
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)


def output_transform(x, clamping=True, pmin=1e-4):
    output = F.log_softmax(x, dim=1)
    if clamping:
        output = torch.clamp(output, np.log(pmin))
    return output


# ----------------------------- deterministic nets -----------------------------
class NNet4l(nn.Module):
    def __init__(self, dropout_prob=0.0, device="cuda"):
        super().__init__()
        self.l1 = Linear(28 * 28, 600, device)
        self.l2 = Linear(600, 600, device)
        self.l3 = Linear(600, 600, device)
        self.l4 = Linear(600, 10, device)
        self.d = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.d(self.l1(x)))
        x = F.relu(self.d(self.l2(x)))
        x = F.relu(self.d(self.l3(x)))
        return output_transform(self.l4(x), clamping=False)


class CNNet4l(nn.Module):
    """4-layer CNN for the MNIST family (architecture control cells)."""

    def __init__(self, dropout_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.d = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        x = F.relu(self.d(self.conv1(x)))
        x = F.relu(self.d(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.d(self.fc1(x)))
        return F.log_softmax(self.fc2(x), dim=1)


class CNNet9l(nn.Module):
    def __init__(self, dropout_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.fcl1 = nn.Linear(4096, 1024)
        self.fcl2 = nn.Linear(1024, 512)
        self.fcl3 = nn.Linear(512, 10)
        self.d = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        x = self.d(F.relu(self.conv1(x)))
        x = self.d(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = self.d(F.relu(self.conv3(x)))
        x = self.d(F.relu(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)
        x = self.d(F.relu(self.conv5(x)))
        x = self.d(F.relu(self.conv6(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.d(self.fcl1(x)))
        x = F.relu(self.d(self.fcl2(x)))
        return F.log_softmax(self.fcl3(x), dim=1)


# ----------------------------- probabilistic nets -----------------------------
class ProbNNet4l(nn.Module):
    def __init__(self, rho_prior, prior_dist="gaussian", device="cuda", init_net=None):
        super().__init__()
        self.l1 = ProbLinear(28 * 28, 600, rho_prior, prior_dist=prior_dist, device=device,
                             init_layer=init_net.l1 if init_net else None)
        self.l2 = ProbLinear(600, 600, rho_prior, prior_dist=prior_dist, device=device,
                             init_layer=init_net.l2 if init_net else None)
        self.l3 = ProbLinear(600, 600, rho_prior, prior_dist=prior_dist, device=device,
                             init_layer=init_net.l3 if init_net else None)
        self.l4 = ProbLinear(600, 10, rho_prior, prior_dist=prior_dist, device=device,
                             init_layer=init_net.l4 if init_net else None)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.relu(self.l3(x, sample))
        return output_transform(self.l4(x, sample), clamping, pmin)

    def compute_kl(self):
        return self.l1.kl_div + self.l2.kl_div + self.l3.kl_div + self.l4.kl_div


class ProbCNNet4l(nn.Module):
    """Probabilistic 4-layer CNN for the MNIST family."""

    def __init__(self, rho_prior, prior_dist="gaussian", device="cuda", init_net=None):
        super().__init__()
        self.conv1 = ProbConv2d(1, 32, 3, rho_prior, prior_dist=prior_dist, device=device,
                                init_layer=init_net.conv1 if init_net else None)
        self.conv2 = ProbConv2d(32, 64, 3, rho_prior, prior_dist=prior_dist, device=device,
                                init_layer=init_net.conv2 if init_net else None)
        self.fc1 = ProbLinear(9216, 128, rho_prior, prior_dist=prior_dist, device=device,
                              init_layer=init_net.fc1 if init_net else None)
        self.fc2 = ProbLinear(128, 10, rho_prior, prior_dist=prior_dist, device=device,
                              init_layer=init_net.fc2 if init_net else None)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x, sample))
        return output_transform(self.fc2(x, sample), clamping, pmin)

    def compute_kl(self):
        return self.conv1.kl_div + self.conv2.kl_div + self.fc1.kl_div + self.fc2.kl_div


class ProbCNNet9l(nn.Module):
    def __init__(self, rho_prior, prior_dist="gaussian", device="cuda", init_net=None):
        super().__init__()
        mk = lambda ic, oc, name: ProbConv2d(ic, oc, 3, rho_prior=rho_prior, prior_dist=prior_dist,
                                             device=device, padding=1,
                                             init_layer=getattr(init_net, name) if init_net else None)
        self.conv1 = mk(3, 32, "conv1")
        self.conv2 = mk(32, 64, "conv2")
        self.conv3 = mk(64, 128, "conv3")
        self.conv4 = mk(128, 128, "conv4")
        self.conv5 = mk(128, 256, "conv5")
        self.conv6 = mk(256, 256, "conv6")
        self.fcl1 = ProbLinear(4096, 1024, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               init_layer=init_net.fcl1 if init_net else None)
        self.fcl2 = ProbLinear(1024, 512, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               init_layer=init_net.fcl2 if init_net else None)
        self.fcl3 = ProbLinear(512, 10, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                               init_layer=init_net.fcl3 if init_net else None)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x, sample))
        x = F.relu(self.conv4(x, sample))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5(x, sample))
        x = F.relu(self.conv6(x, sample))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fcl1(x, sample))
        x = F.relu(self.fcl2(x, sample))
        x = self.fcl3(x, sample)
        return output_transform(x, clamping, pmin)

    def compute_kl(self):
        return (self.conv1.kl_div + self.conv2.kl_div + self.conv3.kl_div + self.conv4.kl_div
                + self.conv5.kl_div + self.conv6.kl_div + self.fcl1.kl_div + self.fcl2.kl_div
                + self.fcl3.kl_div)


# --------------------------------------------------------------------------
# Deeper CIFAR-10 architectures.
#
# These are the 13- and 15-layer networks the reference uses for its tighter
# CIFAR-10 certificates, ported from third_party/PBB/pbb/models.py so that a
# matched comparison against its published CIFAR-10 numbers is possible. The
# layer widths, kernel sizes, padding and pooling positions follow the reference
# exactly; only the construction style matches the rest of this module.
#
# Both consume 32x32x3 and pool four times, so the flattened width is 512*2*2.
# --------------------------------------------------------------------------

_C13 = [(3, 32), (32, 64), (64, 128), (128, 128), (128, 256), (256, 256),
        (256, 256), (256, 512), (512, 512), (512, 512)]
_C15 = [(3, 32), (32, 64), (64, 128), (128, 128), (128, 256), (256, 256),
        (256, 256), (256, 256), (256, 512), (512, 512), (512, 512), (512, 512)]
# indices (1-based, into the conv list) after which a 2x2 max-pool is applied
_P13 = {2, 4, 7, 10}
_P15 = {2, 4, 8, 12}


class _DeepCNNet(nn.Module):
    """Deterministic 13-/15-layer CIFAR-10 network used to learn the prior."""

    def __init__(self, channels, pools, dropout_prob=0.0):
        super().__init__()
        self._pools = pools
        self._n = len(channels)
        for i, (ic, oc) in enumerate(channels, start=1):
            setattr(self, "conv%d" % i, nn.Conv2d(ic, oc, 3, padding=1))
        self.fcl1 = nn.Linear(2048, 1024)
        self.fcl2 = nn.Linear(1024, 512)
        self.fcl3 = nn.Linear(512, 10)
        # The reference uses element-wise Dropout for the 13- and 15-layer nets,
        # not the channel-wise Dropout2d it uses for the 4- and 9-layer ones.
        # With dropout_prob=0.2 on a learnt prior the two differ, so keep it exact.
        self.d = nn.Dropout(dropout_prob)

    def forward(self, x):
        for i in range(1, self._n + 1):
            x = F.relu(self.d(getattr(self, "conv%d" % i)(x)))
            if i in self._pools:
                x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.d(self.fcl1(x)))
        x = F.relu(self.d(self.fcl2(x)))
        return F.log_softmax(self.fcl3(x), dim=1)


class _ProbDeepCNNet(nn.Module):
    """Probabilistic counterpart of :class:`_DeepCNNet`."""

    def __init__(self, channels, pools, rho_prior, prior_dist="gaussian",
                 device="cuda", init_net=None):
        super().__init__()
        self._pools = pools
        self._n = len(channels)
        for i, (ic, oc) in enumerate(channels, start=1):
            name = "conv%d" % i
            setattr(self, name, ProbConv2d(
                ic, oc, 3, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
                padding=1, init_layer=getattr(init_net, name) if init_net else None))
        self.fcl1 = ProbLinear(2048, 1024, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl1 if init_net else None)
        self.fcl2 = ProbLinear(1024, 512, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl2 if init_net else None)
        self.fcl3 = ProbLinear(512, 10, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl3 if init_net else None)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        for i in range(1, self._n + 1):
            x = F.relu(getattr(self, "conv%d" % i)(x, sample))
            if i in self._pools:
                x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fcl1(x, sample))
        x = F.relu(self.fcl2(x, sample))
        return output_transform(self.fcl3(x, sample), clamping, pmin)

    def compute_kl(self):
        kl = self.fcl1.kl_div + self.fcl2.kl_div + self.fcl3.kl_div
        for i in range(1, self._n + 1):
            kl = kl + getattr(self, "conv%d" % i).kl_div
        return kl


class CNNet13l(_DeepCNNet):
    def __init__(self, dropout_prob=0.0):
        super().__init__(_C13, _P13, dropout_prob)


class CNNet15l(_DeepCNNet):
    def __init__(self, dropout_prob=0.0):
        super().__init__(_C15, _P15, dropout_prob)


class ProbCNNet13l(_ProbDeepCNNet):
    def __init__(self, rho_prior, prior_dist="gaussian", device="cuda", init_net=None):
        super().__init__(_C13, _P13, rho_prior, prior_dist, device, init_net)


class ProbCNNet15l(_ProbDeepCNNet):
    def __init__(self, rho_prior, prior_dist="gaussian", device="cuda", init_net=None):
        super().__init__(_C15, _P15, rho_prior, prior_dist, device, init_net)


# model name -> (deterministic class, probabilistic class); CIFAR-10 only
_DEEP = {"cnn13": (CNNet13l, ProbCNNet13l), "cnn15": (CNNet15l, ProbCNNet15l)}


def build_prior_net(model: str, dataset: str, dropout_prob: float, device: str):
    """Construct the deterministic network used to learn / initialise the prior."""
    if model in _DEEP:
        if dataset != "cifar10":
            raise ValueError("%s is a 32x32x3 CIFAR-10 architecture, not %s" % (model, dataset))
        return _DEEP[model][0](dropout_prob=dropout_prob).to(device)
    if model == "cnn":
        if dataset == "cifar10":
            return CNNet9l(dropout_prob=dropout_prob).to(device)
        return CNNet4l(dropout_prob=dropout_prob).to(device)
    return NNet4l(dropout_prob=dropout_prob, device=device).to(device)


def build_prob_net(model: str, dataset: str, rho_prior: float, prior_dist: str, device: str, init_net):
    """Construct the probabilistic network whose posterior is certified."""
    if model in _DEEP:
        if dataset != "cifar10":
            raise ValueError("%s is a 32x32x3 CIFAR-10 architecture, not %s" % (model, dataset))
        return _DEEP[model][1](rho_prior, prior_dist=prior_dist, device=device,
                               init_net=init_net).to(device)
    if model == "cnn":
        if dataset == "cifar10":
            return ProbCNNet9l(rho_prior, prior_dist=prior_dist, device=device, init_net=init_net).to(device)
        return ProbCNNet4l(rho_prior, prior_dist=prior_dist, device=device, init_net=init_net).to(device)
    return ProbNNet4l(rho_prior, prior_dist=prior_dist, device=device, init_net=init_net).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
