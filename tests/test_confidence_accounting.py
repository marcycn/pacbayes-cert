"""P0-5 / DoD: joint confidence is computed by a union bound over delta_mc, delta_pac."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pacbayes_cert.certificates import Certificate, compute_certificate, inv_kl


class _FakeNet:
    def compute_kl(self):
        import torch
        return torch.tensor(10.0)


class _Pb:
    n_bound = 30000
    pmin = 1e-5
    device = "cpu"


def test_joint_confidence_union_bound():
    cert = compute_certificate(
        _Pb(), _FakeNet(), bound_loader=None, mc_samples=10000,
        delta_mc=0.005, delta_pac=0.005, precomputed_mc=(0.02, 0.03),
    )
    assert abs(cert.joint_confidence - 0.99) < 1e-12
    # NOT 1 - delta (=0.995) and NOT 1 - 2*0.01: the two deltas are independent
    assert cert.delta_mc == 0.005 and cert.delta_pac == 0.005


def test_separate_deltas_change_terms():
    # a smaller delta_pac (tighter PAC failure prob) must not decrease the certificate
    base = compute_certificate(_Pb(), _FakeNet(), None, 10000, 0.005, 0.005, precomputed_mc=(0.02, 0.03))
    tighter = compute_certificate(_Pb(), _FakeNet(), None, 10000, 0.005, 0.001, precomputed_mc=(0.02, 0.03))
    # smaller delta_pac => higher confidence => looser (>=) certificate and higher joint conf
    assert tighter.risk_01 >= base.risk_01 - 1e-9
    assert tighter.joint_confidence > base.joint_confidence


def test_inv_kl_monotone():
    # larger slack -> larger upper bound
    assert inv_kl(0.1, 0.01) <= inv_kl(0.1, 0.05)
    # bound is always >= empirical
    assert inv_kl(0.1, 0.02) >= 0.1
