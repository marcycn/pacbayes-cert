#!/usr/bin/env python3
"""Idempotently add a fashion-mnist branch to pbb/data.py::loaddataset."""
from pathlib import Path
f = Path("/root/autodl-tmp/pacbayes-cert/third_party/PBB/pbb/data.py")
s = f.read_text()
branch = """    elif name == 'fashion-mnist':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        train = datasets.FashionMNIST(
            'mnist-data/', train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(
            'mnist-data/', train=False, download=True, transform=transform)
"""
if "name == 'fashion-mnist'" not in s:
    anchor = "    elif name == 'cifar10':"
    assert anchor in s, "cifar10 anchor not found"
    f.write_text(s.replace(anchor, branch + anchor, 1))
    print("PATCHED")
else:
    print("ALREADY_PATCHED")
