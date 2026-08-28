# pacbayes-cert: auditable PAC-Bayes risk certificates

A clean, tested reimplementation of PAC-Bayes-with-Backprop (PBB,
Pérez-Ortiz et al., 2021) that computes **non-vacuous risk certificates** for
neural-network classifiers on MNIST, Fashion-MNIST and CIFAR-10, with corrected
sample accounting, honest joint confidence, batching-invariant Monte-Carlo
aggregation and genuine multi-seed control.

## What this fixes relative to the reference implementation

| Issue | Correction | Test |
|---|---|---|
| Bound-set size used the full dataset length | `n_bound = \|S\S0\|` from explicit indices | `tests/test_split_sizes.py` |
| Prior/posterior/bound partition under-specified | explicit `SplitInfo`, asserted disjoint | `tests/test_split_independence.py` |
| Single `delta` reused, reported as 99% | `delta_mc + delta_pac`, union bound → 0.99 | `tests/test_confidence_accounting.py` |
| Monte-Carlo `/= batch_id` (off-by-one) | example-weighted, batching-invariant | `tests/test_mc_aggregation.py` |
| `--seed` was metadata only | 5 independent seed streams from one base | `tests/test_seed_control.py` |
| Small-data subset was a prefix | uniform random permutation, never reads labels | `tests/test_split_independence.py` |

Class-stratified sampling also removes the prefix bias and was the first fix
tried here, but it was **rejected and removed**: it draws S0 against class counts
taken over all of S, which contains the bound set, so the prior comes to depend
on the bound set's labels, and a sample with fixed class quotas is not the
i.i.d. draw the bound assumes. `make_split` now *raises* on `stratified=True` for
a learnt prior rather than ignoring it, and `SplitInfo.stratified` is always
`False`.

Fifteen data-free-prior runs still carry `stratified: true` in their stored
config, from an older config generation. The flag was never honoured, and there
is no prior/bound split for it to have applied to on those cells — but a flag
cannot establish that either way. `scripts/verify_splits.py` recomputes all 115
saved partitions from their recorded seeds and compares index-for-index against
`split_indices.npz`: **115 recomputed, 0 mismatched**. That, not the flag, is what
shows every reported certificate was evaluated on a label-independent bound set.

Two further corrections go beyond the reference rather than fixing it:

| Step | Why | Test |
|---|---|---|
| Union over candidate priors, `delta_pac / K` with `K = 999` | sigma_0 was chosen by comparing certificates computed on the bound set, so the prior was selected using S_b | `tests/test_prior_selection_union.py` |
| Separate `seed_eval` for the deployment metrics | otherwise the stochastic and ensemble test errors continue the certificate's RNG state and move with `mc_samples` | `tests/test_seed_control.py` |

`K = 999` is not the number of scales we tried. 0.03 was adopted *after* seeing
the curves at {0.01, 0.02, 0.05}, so a set of four is itself a function of the
bound set and a union over it covers nothing. The grid unioned over is instead
every three-decimal scale in (0, 1). It costs log(999) = 6.91 nats: about 0.0005
of certificate on the headline cells, 0.004 on the smallest bound set.

**What that does not settle.** The union is valid for any grid fixed
independently of the bound set, and then covers the adopted scale however it was
picked from that grid. But this grid was written down after the fact, so we
cannot claim the selection was confined to it in advance — a procedure returning
0.0325 would not have been covered. The certificates hold under the assumption
that only a three-decimal scale could have been produced, which the paper carries
as a threat to validity, not as a solved problem.

No union is needed for the posterior: PAC-Bayes-kl is already simultaneous over
all Q, so the learning rate and epoch count may depend on the data freely. That
licence covers the PAC-Bayes event only, not the finite-Monte-Carlo step — so
selecting a checkpoint by comparing finite-m certificates would need its own
correction. This study does not select checkpoints: each certificate is computed
once, on the final-epoch posterior.

Certificates apply to the **Gibbs classifier** only; posterior-mean and ensemble
test errors are reported as descriptive deployment metrics.

The convergence screen that decides whether a learnt-prior run is usable reads
the prior network's 0–1 error on **S0**, the subset it was trained on — a
training diagnostic that consults no held-out data, so it cannot act as a filter
on results. It is applied in `scripts/aggregate.py`.

## Install

```bash
pip install -r requirements-lock.txt   # torch/torchvision matching your CUDA
export PYTHONPATH=$PWD/src
```

## Run

```bash
# one cell
python run.py --config configs/mnist_fquad_learnt_fcn.yaml --base-seed 0 \
    --data-root data --outdir results/raw/mnist_fquad_learnt_fcn_seed0

# the headline matrix over 5 seeds (parallel, resumable)
MAXJOBS=14 SEEDS="0 1 2 3 4" CONFIGS="$(ls configs|sed s/.yaml//)" bash scripts/run_matrix.sh

# unit tests
pytest tests/
```

On a fresh clone this reports 40 passed and 2 skipped: the two tensor-cache
equivalence tests need MNIST on disk and skip until it has been downloaded.
Run anything that touches the data once -- the single-run command above will
do -- and the full 42 run. No test needs a GPU.

## Aggregate, figures, tables

```bash
python scripts/prior_fit_on_s0.py              # prior-net error on S0, for the convergence screen
python scripts/verify_splits.py                # recompute every saved partition; non-zero exit on any mismatch
python scripts/aggregate.py                    # results/raw/* -> results/processed/{runs,summary,registry}.csv
python scripts/make_figures.py                 # registry -> results/figures/*.pdf  (no hard-coded numbers)
python scripts/gen_tables.py                   # registry -> paper/muthesis/generated/*.tex
python scripts/mc_sensitivity.py --run-dir results/raw/<cell>_seed0 --with-analytic
python scripts/gibbs_per_class.py --run-dir results/raw/<cell>_seed0
python scripts/recompute_deployment_metrics.py # bring pre-seed_eval runs onto the current convention
```

`aggregate.py` needs `prior_fit_on_s0.py` to have run first: a learnt-prior run
with no `prior_net_s0_01` field is registered as `unscreened` and excluded rather
than silently admitted.

## Layout

```
src/pacbayes_cert/   clean package (splits, seeds, certificates, models, runner, schema, provenance)
configs/             one YAML per headline cell
scripts/             orchestration, aggregation, figures, tables, MC sensitivity
tests/               unit tests for each audited fix
results/raw/         per-run artifacts (config.resolved.yaml, split_indices.npz,
                     metrics.json, environment.json); the *.pt checkpoints are
                     gitignored and are not published
results/processed/   registry generated by aggregate.py
paper/muthesis/      dissertation (LuaLaTeX + biber); numbers/tables/figures from the registry
third_party/PBB/     vendored upstream (audit reference); patches/PROVENANCE.md documents changes
legacy_invalidated/  pre-correction outputs, retained for audit only — not cited
```


## Provenance

Each run records library versions, GPU, git commit, config hash, all seeds, the
split hash and full sample accounting. The registry carries each certificate
both with and without the prior-selection union (`cert_risk_01` and
`cert_risk_01_nounion`), so the size of that correction is readable rather than
asserted. See `patches/PROVENANCE.md` for the relationship to the upstream PBB
code and the exact list of corrections.
