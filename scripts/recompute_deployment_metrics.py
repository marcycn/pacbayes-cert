"""Recompute the three descriptive test-set metrics under the dedicated seed_eval.

The stochastic and ensemble test errors sample posterior weights.  In the
original ordering they ran straight after the certificate and simply continued
its RNG state, so their value depended on ``mc_samples`` and on how the bound set
had been batched -- raising the Monte-Carlo budget would have moved a test error
that has nothing to do with the budget.  :mod:`pacbayes_cert.runner` now reseeds
from ``seed_eval`` first.

This script brings the already-computed runs onto that convention without
retraining, by reloading each ``posterior.pt`` and re-evaluating on the test set.
The posterior-mean error uses no randomness and is recomputed as a cross-check:
it must come back bit-identical, and a mismatch means the checkpoint and the
recorded metrics have drifted apart.

    python scripts/recompute_deployment_metrics.py [--results results/raw] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pacbayes_cert import data as datamod                       # noqa: E402
from pacbayes_cert.models import build_prob_net                  # noqa: E402
from pacbayes_cert.predictors import (test_ensemble,             # noqa: E402
                                      test_posterior_mean, test_stochastic)
from pacbayes_cert.seeds import SeedBundle, set_deterministic    # noqa: E402

PMIN = 1e-5
ENSEMBLE_SAMPLES = 100


def recompute(run_dir: str, cache: dict, device: str) -> dict:
    m = json.load(open(os.path.join(run_dir, "metrics.json"), encoding="utf-8"))
    ds = m["dataset"]
    if ds not in cache:
        cache[ds] = datamod.load_dataset(ds, root="data")
    _, test = cache[ds]
    loader = torch.utils.data.DataLoader(test, batch_size=250, shuffle=False,
                                         num_workers=0)

    import math
    rho_prior = math.log(math.exp(m["sigma_prior"]) - 1.0)
    net = build_prob_net(m["model"], ds, rho_prior, "gaussian", device, None)
    net.load_state_dict(torch.load(os.path.join(run_dir, "posterior.pt"),
                                   map_location=device))

    seeds = SeedBundle.from_base(m["base_seed"])
    torch.manual_seed(seeds.seed_eval)
    if device == "cuda":
        torch.cuda.manual_seed_all(seeds.seed_eval)

    _, stoch = test_stochastic(net, loader, PMIN, device=device)
    _, postmean = test_posterior_mean(net, loader, PMIN, device=device)
    _, ens = test_ensemble(net, loader, PMIN, device=device, samples=ENSEMBLE_SAMPLES)
    return m, stoch, postmean, ens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/raw")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The runs being reproduced were made under cudnn.deterministic=True; without
    # matching that here, cuDNN picks algorithms by benchmarking and the deeper
    # CIFAR networks land a borderline test example on the other side of an
    # argmax. Observed cost of omitting this: four of 115 runs shifted their
    # posterior-mean error by exactly 1e-4, i.e. one example in 10,000.
    set_deterministic(True)
    cache: dict = {}
    n = 0
    drift = []
    for name in sorted(os.listdir(args.results)):
        d = os.path.join(args.results, name)
        mp = os.path.join(d, "metrics.json")
        if not os.path.isfile(mp) or not os.path.isfile(os.path.join(d, "posterior.pt")):
            continue
        m, stoch, postmean, ens = recompute(d, cache, device)
        dpm = abs(postmean - m["test_postmean_01"])
        if dpm > 1e-9:
            drift.append((name, dpm))
        print("%-46s stoch %.4f (was %.4f)  ens %.4f (was %.4f)  postmean d=%.2e"
              % (name, stoch, m["test_stoch_01"], ens, m["test_ens_01"], dpm))
        if not args.dry_run:
            # Keep the first as-run value if this script is run twice; otherwise a
            # second pass would record its own output as the original.
            m.setdefault("test_stoch_01_asrun", m["test_stoch_01"])
            m.setdefault("test_ens_01_asrun", m["test_ens_01"])
            m["test_stoch_01"] = round(float(stoch), 6)
            m["test_ens_01"] = round(float(ens), 6)
            m["test_postmean_01"] = round(float(postmean), 6)
            m["seed_eval_applied"] = True
            with open(mp, "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2)
        n += 1
    print("\n%d runs re-evaluated" % n)
    if drift:
        print("!! posterior-mean drift on %d runs (checkpoint/metrics mismatch):" % len(drift))
        for name, dd in drift[:10]:
            print("   %-46s %.3e" % (name, dd))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
