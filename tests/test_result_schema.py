"""P2 §5.9 / DoD: typed result schema round-trips and is versioned."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pacbayes_cert.schema import RunResult, SCHEMA_VERSION


def test_roundtrip():
    r = RunResult(run_id="x", dataset="mnist", n_bound=30000, cert_risk_01=0.04552)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "metrics.json")
        r.to_json(p)
        r2 = RunResult.from_json(p)
    assert r2.run_id == "x"
    assert r2.n_bound == 30000
    assert abs(r2.cert_risk_01 - 0.04552) < 1e-12
    assert r2.schema_version == SCHEMA_VERSION


def test_distinct_metric_fields_exist():
    r = RunResult(run_id="x")
    for f in ("mc_err_01", "emp_risk_01", "cert_risk_01", "test_stoch_01",
              "test_postmean_01", "test_ens_01"):
        assert hasattr(r, f), f"missing distinct metric field {f}"
