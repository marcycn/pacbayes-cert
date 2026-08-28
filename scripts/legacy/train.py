#!/usr/bin/env python3
"""Run ONE PBB experiment config (wraps pbb.utils.runexp), capture the ***Final results***
CSV line, and write metadata.json + metrics.json + stdout.txt to <outdir>.

Designed to be config/grid-friendly: call once per cell. Aggregate with aggregate_results.py.
PYTHONPATH must include the PBB repo root (third_party/PBB).
"""
import argparse, contextlib, io, json, os, pathlib, sys, time

HEADER = ["Objective","Dataset","Sigma","pmin","LR","momentum","LR_prior","momentum_prior",
          "kl_penalty","dropout","Obj_train","Risk_CE","Risk_01","KL","Train NLL loss",
          "Train 01 error","Stch loss","Stch 01 error","Post mean loss","Post mean 01 error",
          "Ens loss","Ens 01 error","01 error prior net","perc_train","perc_prior"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)          # mnist / fashion-mnist / cifar10
    ap.add_argument("--objective", required=True)        # fquad / flamb / fclassic / bbb
    ap.add_argument("--prior", required=True)            # rand / learnt
    ap.add_argument("--model", required=True)            # fcn / cnn
    ap.add_argument("--mc-samples", type=int, default=1000)
    ap.add_argument("--train-epochs", type=int, default=10)
    ap.add_argument("--prior-epochs", type=int, default=70)
    ap.add_argument("--perc-train", type=float, default=1.0)
    ap.add_argument("--perc-prior", type=float, default=0.5)
    ap.add_argument("--sigma-prior", type=float, default=0.03)
    ap.add_argument("--pmin", type=float, default=1e-5)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--momentum", type=float, default=0.95)
    ap.add_argument("--lr-prior", type=float, default=0.005)
    ap.add_argument("--momentum-prior", type=float, default=0.99)
    ap.add_argument("--kl-penalty", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.025)
    ap.add_argument("--delta-test", type=float, default=0.01)
    ap.add_argument("--layers", type=int, default=9)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")  # headless safety (pbb imports make_grid)
    import numpy as np, torch
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    from pbb.utils import runexp

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        runexp(args.dataset, args.objective, args.prior, args.model, args.sigma_prior, args.pmin,
               args.lr, args.momentum, learning_rate_prior=args.lr_prior,
               momentum_prior=args.momentum_prior, delta=args.delta, layers=args.layers,
               delta_test=args.delta_test, mc_samples=args.mc_samples, kl_penalty=args.kl_penalty,
               train_epochs=args.train_epochs, device=device, prior_epochs=args.prior_epochs,
               dropout_prob=args.dropout, perc_train=args.perc_train, perc_prior=args.perc_prior,
               batch_size=250, verbose=False)
    dt = time.time() - t0
    out = buf.getvalue()
    (pathlib.Path(args.outdir) / "stdout.txt").write_text(out)

    metrics = None
    lines = out.splitlines()
    for i, l in enumerate(lines):
        if "***Final results***" in l:
            data_line = lines[i + 2]
            fields = [f.strip() for f in data_line.split(",")]
            metrics = dict(zip(HEADER, fields))
            break

    meta = dict(vars(args)); meta.update(device=str(device), duration_sec=round(dt, 1),
                                         torch_version=torch.__version__,
                                         cuda_available=torch.cuda.is_available())
    (pathlib.Path(args.outdir) / "metadata.json").write_text(json.dumps(meta, indent=2))
    if metrics:
        metrics["duration_sec"] = round(dt, 1)
        (pathlib.Path(args.outdir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print("OK " + json.dumps({"run_id": os.path.basename(args.outdir),
                                  "Risk_CE": metrics["Risk_CE"], "Risk_01": metrics["Risk_01"],
                                  "Stch_01": metrics["Stch 01 error"],
                                  "PostMean_01": metrics["Post mean 01 error"],
                                  "Ens_01": metrics["Ens 01 error"],
                                  "duration_sec": round(dt, 1)}))
    else:
        print("WARNING: no ***Final results*** line parsed; inspect stdout.txt", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
