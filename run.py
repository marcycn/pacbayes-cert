#!/usr/bin/env python3
"""CLI entry point: run one experiment cell from a YAML config + overrides.

Examples
--------
    python run.py --config configs/mnist_fquad_learnt_fcn.yaml --base-seed 0 \
        --outdir results/raw/mnist_fquad_learnt_fcn_seed0

    python run.py --config configs/mnist_fquad_learnt_fcn.yaml --base-seed 1 \
        --train-epochs 2 --mc-samples 1000 --outdir /tmp/smoke
"""
import argparse
import io
import json
import os
import sys
import contextlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from pacbayes_cert.config import ExpConfig  # noqa: E402
from pacbayes_cert.runner import run_experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config; omit to use defaults")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--verbose", action="store_true")
    # overrides (None = keep config value)
    for name, typ in [
        ("dataset", str), ("model", str), ("objective", str), ("prior-type", str),
        ("base-seed", int), ("perc-train", float), ("perc-prior", float),
        ("label-noise", float), ("sigma-prior", float), ("kl-penalty", float),
        ("lr", float), ("lr-prior", float), ("train-epochs", int), ("prior-epochs", int),
        ("batch-size", int), ("mc-samples", int), ("delta-train", float),
        ("delta-mc", float), ("delta-pac", float), ("device", str),
    ]:
        ap.add_argument(f"--{name}", type=typ, default=None)
    args = ap.parse_args()

    cfg = ExpConfig.from_yaml(args.config) if args.config else ExpConfig()
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config", "outdir", "data_root", "verbose") and v is not None}
    cfg = cfg.merged(**overrides)

    os.makedirs(args.outdir, exist_ok=True)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = run_experiment(cfg, args.outdir, data_root=args.data_root, verbose=args.verbose)
    finally:
        (open(os.path.join(args.outdir, "stdout.log"), "w")).write(buf.getvalue())

    print("OK " + json.dumps({
        "run_id": result.run_id, "n_bound": result.n_bound, "n_posterior": result.n_posterior,
        "cert_risk_01": round(result.cert_risk_01, 5), "kl": round(result.kl, 4),
        "joint_confidence": result.joint_confidence, "test_stoch_01": round(result.test_stoch_01, 5),
        "duration_sec": result.duration_sec,
    }))


if __name__ == "__main__":
    main()
