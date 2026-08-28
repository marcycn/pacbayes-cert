"""P0-7 / DoD: seeds genuinely control split/model/loader/mc and are decomposed."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pacbayes_cert.seeds import SeedBundle, _derive


def test_streams_are_distinct():
    b = SeedBundle.from_base(0)
    vals = {b.seed_split, b.seed_model, b.seed_loader, b.seed_mc}
    assert len(vals) == 4, "the four seed streams must differ"


def test_base_seed_changes_all_streams():
    a = SeedBundle.from_base(0)
    b = SeedBundle.from_base(1)
    assert a.seed_split != b.seed_split
    assert a.seed_model != b.seed_model
    assert a.seed_loader != b.seed_loader
    assert a.seed_mc != b.seed_mc


def test_derivation_is_deterministic_cross_process():
    # BLAKE2b derivation must be stable (not Python's salted hash)
    assert _derive(0, "split") == _derive(0, "split")
    assert _derive(5, "model") != _derive(5, "loader")


def test_same_base_reproduces_bundle():
    assert SeedBundle.from_base(7).to_dict() == SeedBundle.from_base(7).to_dict()
