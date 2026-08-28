"""Training loops for the deterministic prior network and the probabilistic posterior."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange


def train_prior_net(net, loader, epochs, lr, momentum, device="cuda", verbose=False):
    """Train the deterministic network used to build the data-dependent prior."""
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    for epoch in trange(epochs, disable=not verbose, desc="prior"):
        net.train()
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            net.zero_grad()
            loss = F.nll_loss(net(data), target)
            loss.backward()
            optimizer.step()
    return net


def train_posterior(net, pbobj, loader, epochs, lr, momentum, lambda_var=None,
                    device="cuda", verbose=False):
    """Train the probabilistic posterior with a PAC-Bayes objective."""
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    optimizer_lambda = None
    if pbobj.objective == "flamb":
        optimizer_lambda = optim.SGD(lambda_var.parameters(), lr=lr, momentum=momentum)
    clamping = pbobj.objective != "bbb"

    for epoch in trange(epochs, disable=not verbose, desc="posterior"):
        net.train()
        if lambda_var is not None:
            lambda_var.train()
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            net.zero_grad()
            obj, _, _, _, _ = pbobj.train_obj(net, data, target, clamping=clamping, lambda_var=lambda_var)
            obj.backward()
            optimizer.step()
            if pbobj.objective == "flamb":
                lambda_var.zero_grad()
                obj_l, _, _, _, _ = pbobj.train_obj(net, data, target, clamping=clamping, lambda_var=lambda_var)
                obj_l.backward()
                optimizer_lambda.step()
    return net
