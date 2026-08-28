# EXPERIMENT_PLAN — PAC-Bayes-with-Backprop certificate study (CCF-B target)

> Tiered so it degrades gracefully: **Tier 0** (reproduction gate) is mandatory and is the project kill-switch; **Tier 1** is the dissertation core; **Tier 2** is the CCF-B-carrying novelty. Lower tiers are subsets of higher ones.
> Code developed locally (`src/`); during the early exploratory phase data and checkpoints lived on a rented remote GPU box (path redacted). The reported study is entirely local. PBB official repo imported as `third_party/PBB`.

## Reproduction targets (JMLR 2021, arXiv:2007.12911 — note brief's 2107.13560 is WRONG)
MNIST Table 1 (cert / stochastic-test-error):
- FCN+Rand.Init: fquad .3155/.0268 · flambda .3275/.0211 · fclassic .3304/.0407 · fbbb .5516/.0088
- FCN+Learnt: fquad .0279/.0084 · flambda .0354/.0082 · fclassic .0284/.0101 · fbbb .0968/.0063
- CNN+Learnt: fquad .0155/.0045 · (flambda/fclassic per table)
- ferm: FCN .0152 · CNN .0092 (test error)
- Table 2 head-to-head: PBB fquad cert .0279 vs Dziugaite-Roy SGLD .2100
- CIFAR-10 Table 5: 9-layer fquad(70%) cert .2377 · 13-layer fquad(70%) .1758 · 15-layer flambda(70%) .2121
> Confirm exact digits from the PDF table before locking the 0.5pp tolerance.

## PBB API surface (from `third_party/PBB`)
`runexp(dataset∈{mnist,cifar10}, method∈{fquad,flamb,fclassic,bbb}, prior∈{rand,…}, arch∈{fcn,cnn}, sigmaprior, pmin, lr, momentum, …, delta, delta_test, mc_samples, train_epochs, perc_train, perc_prior, prior_epochs, layers)`.
- **Fashion-MNIST is NOT native** → add an `elif name=='fashion-mnist'` branch to `pbb/data.py` (torchvision `datasets.FashionMNIST`, same 28×28 transform as MNIST). Trivial drop-in.
- `perc_train` → small-data regime (Tier 2 N4). `prior='rand'` vs `perc_prior`+learnt → prior axis. `mc_samples` (paper=150k, hours) → the compute lever.

## Tier 0 — Reproduction gate (KILL-SWITCH; ~week 1)
Goal: prove the implementation is correct before any extension.
- Wrap `runexp` behind `src/train.py --config configs/*.yaml`; one `metadata.json` per run (seed, config hash, git/patch id, durations).
- Reproduce MNIST Table 1: FCN {fquad,flamb,fclassic,fbbb}×{Rand.Init,Learnt}; CNN {fquad,flamb,fclassic}×{Learnt}; ferm baseline. ≥3 seeds (5 for cells we publish).
- **fquad numerical-stabilization budget** (safe softmax/kl forms) — explicitly tracked; fquad is the touchy one.
- mc_samples: 1k–10k for dev/trend; **150k for final certificates on key cells only**.
- **Decision gate**: match Table 1 within 0.5pp (cert + test error). If not achievable in ~10 days → escalate/reconsider before FMNIST/CIFAR.

## Tier 1 — Cross-dataset ladder (dissertation core)
Goal: MNIST → Fashion-MNIST → CIFAR-10 as a 3-rung difficulty ladder.
- Add Fashion-MNIST; run the Tier-0 grid (subset: fquad/flamb/fclassic, Rand+Learnt, FCN+CNN) on all three.
- CIFAR-10: 9-layer CNN (official) primary; 13/15-layer **optional** (compute-risk).
- Report per cell: test error, accuracy, NLL, ECE, Brier, PAC-Bayes cert (ce & 01), **slack = cert − emp risk**.
- Pre-registered falsifier (good-question Card 1): if {fquad,flamb,fclassic} slack ordering is identical across the 3 datasets with effect-size change <15% → pivot to a **stability** claim instead of a "flip" claim.

## Tier 2 — CCF-B-carrying novelty (the pivot; see ADVERSARIAL_REVIEW.md)
- **N1 Deployed-predictor certificate (THE contribution).** Compute certificates for posterior-mean & majority-vote predictors, not just Gibbs. Adapt **Lacasse 2006 Cq/Ct** (Gibbs mean+var → majority-vote risk) to fquad/flambda. Define **deployed-cert gap** = cert(deployed) − true deployed-predictor risk; measure across the ladder. Falsifier: if gap is trivially small everywhere → the validity question is moot (reportable negative).
- **N2 Exogenous difficulty.** Difficulty metrics from raw data only: injected-label-noise rate, k-NN class-overlap, intrinsic-dimension proxy. Regress slack on difficulty; **difficulty×objective interaction test** with bootstrap CIs.
- **N3 3-axis interaction cell**: objective × predictor-rule × prior (the empty cell no prior paper fills), 5 seeds + bootstrap CIs.
- **N4 Small-data crossover (Card 2, secondary)**: self-certified vs hold-out, n∈{500,1000,2000,5000}; locate crossover n* per dataset with CIs. Position as quantifying the 2111.07737 regime boundary.
- **N5 Prior-transfer causal validation (Card 3, secondary)**: decoy/label-permuted-source control applied to 2109.10304-style transfer; do NOT claim first-ness.
- **Analytical hook**: KL-term vs empirical-risk-term slack decomposition as a function of exogenous difficulty.

## Metrics, statistics, reproducibility
- Dependent: test error, acc, NLL, ECE, Brier, cert_ce, cert_01, slack, deployed-cert gap, (KL, emp-risk) decomposition, wall-clock, peak VRAM.
- Independent: objective, prior type, predictor rule, dataset, train fraction, architecture.
- **5 seeds** for any published ordering/interaction claim; mean±std + **95% bootstrap CIs**; paired **Wilcoxon signed-rank** for key pairwise comparisons; falsifiers are **CI-non-overlap** tests, not point thresholds.
- **No data reuse**: model-selection data ≠ final independent-test reporting; the bound-set partition is fixed per the PBB protocol.
- Registry: every run → `experiments/logs/<run>/metadata.json` + metrics; `aggregate_results.py` → `results/results_summary.csv`; **every paper number traceable to a run id**; failed runs logged (no cherry-picking).

## Compute budget (1× NVIDIA TITAN Xp, 12 GB)
- MNIST/FMNIST FCN+LeNet-CNN: cheap (minutes/seed).
- CIFAR-10 9-layer: moderate; 13/15-layer optional/risky.
- **mc_samples=150k is the cost driver** (hours/cert) → reserve for final certs on published cells; 1k–10k for dev/gate.
- Architectures kept at LeNet/compact-CNN scale (matches JMLR; no ResNets) — fits 12 GB comfortably.

## Deliverables mapping
- Tier 0 → `REPRO_LOG.md` + reproduced Table 1 (also the MSc deliverable "replication on MNIST").
- Tier 1 → Fashion-MNIST new code+results + CIFAR-10 analysis (MSc deliverables).
- Tier 2 → the CCF-B manuscript's novel core.
- All → `paper/main.tex` (LaTeX, local), figures from `results_summary.csv` only.
