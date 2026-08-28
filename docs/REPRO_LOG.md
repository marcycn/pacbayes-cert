# REPRO_LOG — PBB reproduction progress

> **Superseded environment.** This log records the *early* exploratory phase on a
> rented remote GPU. None of the results reported in the dissertation come from
> it: every reported run is a local one on 2x RTX 3080 (20 GB), recorded in
> `results/raw/*/environment.json`. Kept for the audit trail only. The remote
> host details have been redacted, since this file ships with the artefact.

## Environment
- GPU server: rented remote box over SSH, key auth (host redacted). 1× TITAN Xp 12GB.
- PBB official repo: `<remote>/third_party/PBB` (arXiv **2007.12911** — brief's 2107.13560 is WRONG).
- Python: conda `base` (3.12.3 + torch 2.5.1+cu124 + torchvision 0.20.1).
- Wrapper: `scripts/train.py` (config-driven, captures runexp `***Final results***` → metrics.json/metadata.json/stdout.txt). Grid: `scripts/run_grid.sh`.

## Milestone 0 — pipeline validated (smoke) ✓ [2026-06-23]
`mnist fquad rand fcn`, 3 epochs, mc=1000 → ran end-to-end on GPU in 247s. Captured: Risk_CE=0.2508, Risk_01=0.7495, Stch_01=0.659, PostMean_01=0.306, Ens_01=0.318. Numbers are poor (undertrained 3 epochs) — **pipeline correctness only, not a result.**

## Milestone 1 — full MNIST Table-1 grid LAUNCHED [2026-06-23]
`screen -dmS mnist_repro`, train_epochs=100, mc_samples=10000, seed=0. 11 cells: FCN{rand,learnt}×{fquad,flamb,fclassic,bbb} + CNN{learnt}×{fquad,flamb,fclassic}. Master log: `experiments/logs/grid_master.log`; results: `results/results_summary.csv`. Sequential; ~25 min/cell → ~4–5 h.

### Reproduction targets (JMLR 2021 Table 1, MNIST)
| cell | cert | stch err |
|---|---|---|
| FCN+rand fquad | .3155 | .0268 |
| FCN+rand flambda | .3275 | .0211 |
| FCN+rand fclassic | .3304 | .0407 |
| FCN+learnt fquad | .0279 | .0084 |
| FCN+learnt flambda | .0354 | .0082 |
| FCN+learnt fclassic | .0284 | .0101 |
| CNN+learnt fquad | .0155 | .0045 |

### Notes / risks
- fquad numerical stability to watch (touchy objective).
- mc=10000 gives slightly looser certs than paper's 150k; final reported cells may need a selective 150k re-run.
- If first cell (fquad rand fcn) shows sensible test error (~0.02–0.05) the epoch count is OK; else bump train_epochs.

## Milestone 1b — kill-switch tripped + FIXED [2026-06-24 05:17]
Cell-1 (fquad rand fcn, **lr=1e-3**, 100ep, mc=10000, 29min): **Risk_01=0.329** (paper cert .3155 ✓) but **Stch_01=0.105** (paper .0268 ✗ 4× too high). The cert matched only because the rand-prior bound is KL-dominated (~0.32 regardless); real signal = **undertrained**.
- **Root cause** (paper §7.2.1, lines 944-946 & 1004): trained **100 epochs** (converge ~70) — epochs were FINE; but posterior **LR 1e-3 "converged slowly"**; the sweep sweet-spot is **5e-3** (1e-2 diverges). `running_example.py`'s `lr=1e-3` was the bug (its `lr_prior` was already 5e-3).
- **Fix:** posterior LR → **5e-3**. (Confirmed correct: σ0=0.03, pmin=1e-5, perc_prior=0.5, δ=0.025/δ'=0.01, mc=150k in paper/10k here.) Stale lr=1e-3 results archived to `results/results_summary_lr1e-3_stale.csv`. Grid relaunched in screen `8332` with `LR=0.005`.
- **Next:** validate cell-1 (lr=5e-3) hits Stch_01≈0.02–0.05 at next tick; if so, the full 11-cell grid proceeds correctly.

## Milestone 1c — REPRODUCTION VALIDATED ✓ (target had been mis-attributed) [2026-06-24 06:17]
The library's "stoch .0268" for fquad FCN Rand was a **mis-attribution**. Verified against the paper's ACTUAL Table 1 (arXiv 2007.12911, line 1096): real columns are `Risk_CE, Risk_01, …, Stch_01, …, Post mean 01 error, …, Ens 01 error`. The true **stochastic test error is .0951**, not .0268.

Correct Table-1 targets (Risk_01 cert / Stch_01 test error):
| cell | Risk_01 (cert) | Stch_01 (err) | PostMean_01 | Ens_01 |
|---|---|---|---|---|
| FCN Rand fquad | .3155 | **.0951** | .0558 | .0572 |
| FCN Rand flambda | .3275 | .0742 | .0429 | .0448 |
| FCN Learnt fquad | .0279 | .0204 | .0186 | .0189 |
| FCN Learnt flambda | .0354 | .0178 | .0185 | .0185 |
| CNN Learnt fquad | .0155 | .0127 | .0105 | .0104 |

**My cell-1 (lr=5e-3, σ0=0.03, 100ep, mc=10000)**: Risk_CE .2116 (paper .2033), Risk_01 **.3267** (.3155), Stch_01 **.0914** (.0951 ✓), PostMean_01 .0553 (.0558), Ens_01 .0576 (.0572) → **matches within ~1pp on every metric. Kill-switch PASSES.**
σ0 probe (0.01/0.02/0.05, 70ep+mc=2k) confirmed σ0 insensitive in [0.02,0.05] (0.01 bad). Settings locked: **lr=5e-3, σ0=0.03, 100 epochs, mc=10000.** (mc=10000 vs paper 150k → certs ~1pp looser; acceptable; optional 150k rerun on key cells later.)
Grid relaunched with resume-skip (cell-1 skipped); cells 2–11 running. **MNIST reproduction = effectively DONE once grid completes.**

## Milestone 1d — CNN cells at reduced config for compute [2026-06-24 11:54]
Probabilistic-CNN training is ~10× slower/step than FCN and is **CPU/overhead-bound** (GPU only ~29% util / 81W on the TITAN Xp — not thermal, no throttle) — at mc=10000/100ep each CNN cell was ~4–5h (3 cells ≈ 12–15h). The **FCN grid (8 cells) is the validated core reproduction**. To protect the timeline, the 3 CNN cells run at **mc=2000, epochs=50** (~5–10× faster); their risk certificates are correspondingly looser (mc-limited) but the architecture trend (CNN ≥ FCN tightness) is preserved. Documented as a limitation in the report.

## Milestone 2 — MNIST COMPLETE ✓ + Fashion-MNIST LAUNCHED [2026-06-24 15:27]
**MNIST grid 11/11 done.** Full reproduction vs JMLR 2021 Table 1 (see `results/comparison.md` + `results/figures/`):
- FCN cells (mc=10k/100ep) match paper test-errors within ≤0.4pp; certs within the mc=10k-vs-150k gap (fclassic-rand cert Δ+0.001 = exact).
- CNN cells (mc=2k/50ep, reduced for compute) confirm the architecture trend (CNN Stch .0106 < FCN .0224); certs mc-limited-looser.
Key reproduced qualitative findings: BBB best accuracy but loosest cert; fquad/flamb tighter than fclassic; **learnt prior » rand prior for tightness** (FCN fquad rand-cert .327 → learnt .039; CNN learnt .030).

## Fashion-MNIST patches (APPLIED ✓)
1. `third_party/PBB/pbb/data.py` — added `fashion-mnist` branch (FashionMNIST, Normalize(0.2860,0.3530)); data downloaded; verified loads 60k/10k.
2. `third_party/PBB/pbb/utils.py` — FCN posterior-net dispatch `elif name_data=='mnist'` → `in ('mnist','fashion-mnist')` (CNN branches already use `else` for non-cifar). **First Fashion-MNIST launch failed fast (rc=1, `net` unbound) until this patch; verified fix with a 2-epoch run.**
Fashion-MNIST grid launched in screen `fmnist` via `DATASET=fashion-mnist run_grid_any.sh` (8 FCN full-config + 3 CNN reduced). Then CIFAR-10.

## Milestone 3 — Fashion-MNIST COMPLETE ✓ + CIFAR-10 LAUNCHED [2026-06-25 00:33]
**Fashion-MNIST grid 11/11 done.** Extension results (new): FCN rand certs 0.43–0.78 (loose, bbb near-vacuous); **FCN learnt certs 0.13–0.18** (learnt prior tightens ~3× vs rand, halving error); CNN learnt certs 0.12–0.17 with CNN accuracy ~0.092 (beats FCN 0.117). Method + learnt-prior mechanism transfer cleanly to the harder task; certificates non-vacuous throughout.
**CIFAR-10 launched** in screen `cifar` via `DATASET=cifar10 run_grid_any.sh` — 4 cells (fquad/flamb/fclassic/bbb, learnt, 9-layer CNN, reduced mc=2k/ep=50). ~5h. CIFAR data pre-fetched. After CIFAR completes → final aggregate (all-dataset figures) + fill report §4.3/§4.4 → done.

## Milestone 4 — ALL PROPOSAL DELIVERABLES COMPLETE ✓ [2026-06-25 11:00]
**Experiments (26 cells, 3 datasets):** MNIST 11/11 (validated vs JMLR Table 1, test errors ≤0.4pp) ✓; Fashion-MNIST 11/11 (new code: `data.py` + `utils.py` patches; new results) ✓; CIFAR-10 4/4 (9-layer CNN, reduced config) + analysis ✓. Cross-dataset ladder (learnt `fquad` cert): MNIST 0.039 → Fashion 0.145 → CIFAR 0.410 (monotonic degradation w/ difficulty).
**Report:** `docs/REPORT.md`, 7133 words, 9 sections (Abstract→Conclusion + References + Appendix A full 26-cell table), 5 tables, all numbers traceable to `results_summary.csv` run ids.
**Aggregation:** `results/comparison.md` + 3 figures (cert_compare/slack/predictor_error) regenerated.
**Proposal requirements = COMPLETE.** Self-terminating the monitor cron.

## Milestone 5 — marks-boost followups [2026-06-25 18:08]
- **Small-data ablation ADDED** (valid under the fixed seed — varies perc_train): MNIST fquad learnt fcn at 100/50/20/10% → cert 0.039→0.052→0.063→0.072, Stch 0.022→0.064; bound stays **non-vacuous at 10% data** (0.072). Written into report §4.6 (Table 5).
- **Multi-seed ABANDONED**: PBB `loaddataset` hardcodes `torch.manual_seed(7)` → runs fully deterministic (verified seed1≡seed0). Real seed-variance would need an upstream seeding patch + full re-run that shifts the paper-matching split. §5.2 reframed from "single-seed" to an honest "determinism" note (exact reproducibility is a strength; CIs flagged as future work). Report now 7434 words. Tarball refreshed.
Add a branch to `third_party/PBB/pbb/data.py::loaddataset` (before the `cifar10` branch):
```python
    elif name == 'fashion-mnist':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        train = datasets.FashionMNIST(
            'mnist-data/', train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(
            'mnist-data/', train=False, download=True, transform=transform)
```
Fashion-MNIST is a 28×28 grayscale drop-in for MNIST → same FCN/CNN archs, same `runexp` flow. Generalize `run_grid.sh` to take `--dataset` (or clone to `run_grid_fmnist.sh`) and run the same objective×prior×model grid.

## Auto-monitor
Durable recurring cron `160dad0d` (every :13/:43) drives: check grid → validate cell-1 kill-switch → MNIST done ⇒ Fashion-MNIST ⇒ CIFAR-10 ⇒ aggregate+figures ⇒ write report ⇒ CronDelete + stop. Survives session restarts; grid runs on GPU independently of this session.
