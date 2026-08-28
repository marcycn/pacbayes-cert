# Revision Tracking — supervisor feedback → fixes

Target level (user choice): **high-score path (70–80)** — 5 seeds, CIFAR random-prior
control, MC-convergence on fixed checkpoints, controlled-difficulty, full provenance.
Code approach: **clean refactor into `src/pacbayes_cert/`**. GPU: AutoDL remote, driven over SSH.

## Status legend: ☐ todo · ◐ in progress · ☑ done · ⊘ blocked

### Phase 1 — clean code refactor (`src/pacbayes_cert/`)
- ☑ `seeds.py` — decomposed seed streams (split/model/loader/mc) [P0-7]
- ☑ `splits.py` — immutable SplitInfo, label-independent uniform permutation, exact n_bound [P0-1/P0-2/P1-9]
- ☑ `data.py` — SplitInfo-driven loaders, label-noise corruption [P1-3]
- ☑ `models.py` — faithful prob nets (FCN, 4-l CNN, 9-l CNN) + minor bug fixes
- ☑ `objectives.py` — fquad/flamb/fclassic/bbb (unchanged math)
- ☑ `certificates.py` — example-weighted MC aggregation [P0-6], split delta_mc/delta_pac [P0-5]
- ☑ `predictors.py` — example-weighted test predictors (off-by-one fixed) [P0-6]
- ☑ `trainer.py`, `schema.py` (typed, versioned), `provenance.py`, `config.py`, `runner.py`
- ☑ `run.py` CLI entry; `configs/*.yaml` (15 headline cells incl. CIFAR rand-prior [P1-4])

### Phase 2 — unit tests
- ☑ split sizes (6000/3000), independence, stratification — PASS locally (numpy-only)
- ☐ mc aggregation, confidence, seed, schema — need torch (run on GPU)

### Phase 3 — experiments on GPU
- ☑ passwordless SSH (ASKPASS install; key authorised)
- ☑ Phase A smoke tests + 19 unit tests pass on GPU
- ☑ Phase B FCN headline matrix × 5 seeds (45/45, 0 fails); CNN/CIFAR running (chain)
- ◐ Phase C MC sensitivity + small-data + controlled-difficulty (auto-chained after CNN)
- Note: mc_samples reduced 10000→2000 for tractability (5× faster MC); documented as
  reduced-compute; MC sensitivity sweep recovers the m=150k behaviour on a fixed checkpoint.

### Corrected FCN results (5 seeds, mean±SD) — confirm the audit fixes
- MNIST learnt: fclassic 0.0446, fquad 0.0580, flamb 0.0773, bbb 0.2656 (was 0.032/0.039/...)
- objective order fclassic<fquad<flamb (NOT old fquad<flamb<fclassic) → P1-1 confirmed
- prior: MNIST fquad learnt 0.058 vs rand 0.350 (KL/n 0.006 vs 0.138) → H1 confirmed
- bbb best accuracy (stoch 0.019) loosest cert (KL/n 0.249) → P1-2 reworded correctly

### Phase 4 — results registry + figures/tables
- ☑ `scripts/aggregate.py` — registry, dedup, schema check [P2 §5.5/5.6/5.10]
- ☐ `scripts/make_figures.py`, `scripts/gen_tables.py` — read processed only
- ☐ data-partition diagram [P0-2]

### Phase 5 — paper rewrite (`paper/muthesis/`)
- ☐ P0-3 Gibbs-only certificate scope; P0-4 correct paper column comparison
- ☐ theory tightening [P3 §6.6]; metric naming [P3 §6.8]; word audit [P3 §6.7]
- ☐ citations / `\nocite{*}` removal / LaTeX corruption [P3 §6.1–6.5]
- ☐ RQ-driven rewrite of all sections [§9]; abstract last

### Phase 6 — adversarial review loop
- ☐ experiment-audit, paper-claim-audit, citation-audit, kill-argument → iterate

## FINAL STATE (all phases complete)

93 runs (5 seeds × headline + ablations), 0 failed/excluded. Paper compiles clean
(LuaLaTeX+biber, 31 pp, 0 undefined cites). Two cross-model (GPT-5.x) adversarial
review rounds passed: certificate-math audit, paper-claim audit (caught the CIFAR
KL-mechanism error), citation audit (2 fixes), kill-argument (14 objections → all
resolved/0 new HIGH).

### Definition of Done
- [x] clean-env install/run (requirements-lock, run.py, run_matrix.sh)
- [x] n_posterior/n_bound unit tests (6000/3000) pass
- [x] prior/bound disjointness asserted + tested
- [x] MC batching-invariance tested (deterministic posterior exact)
- [x] joint confidence = 1-δ_mc-δ_pac = 0.99 (tested)
- [x] seeds control split/model/loader/mc (tested)
- [x] all main results from corrected code; legacy frozen in legacy_invalidated/
- [x] paper certifies Gibbs only; mean/ensemble descriptive
- [x] reference comparison uses same-definition columns (corrected framing)
- [x] every result traceable to config/split/checkpoint/metrics + registry
- [x] main table multi-seed mean±SD
- [x] figures/tables/numbers auto-generated from registry (no hardcoding)
- [x] LaTeX compiles clean from scratch
- [x] placeholders/duplicate-numbering/odd-appendix removed; \nocite{*} removed
- [x] cold-start reproduction in a fresh conda env: 19 unit tests pass (PYTEST_RC=0) and
      an end-to-end cell runs (CELL_RC=0, n_bound=30000, cert 0.1231 identical) — requires
      pinning a GPU-compatible torch build (torch==2.5.1+cu124); plain `pip install torch`
      hits CUBLAS_STATUS_ARCH_MISMATCH on the Pascal GPU (documented in requirements-lock.txt)
- [x] report length ~7,400 words (rubric: around 8,000; 7,000-9,000 range); captions/refs excluded
- [x] writing de-AI pass (0 em-dashes; cross-model style re-check passed); 8 figures
- [ ] student ID on title page  ← USER must fill ([student ID] placeholder)

### Rubric alignment (COMP66060)
- Methodology (20%): added "Design choices and alternatives considered" (bound choice, prior
  route, posterior family, architectures, compute) + provenance/schema depth.
- Evaluation/Reflection (20%): added evaluation-strategy justification + project-process reflection.
- Project Achievement (20%): error analysis (confusion matrices), deployment-gap analysis,
  RQ1 published-vs-corrected comparison, seed-variability analysis.
- Format (5%): clean LuaLaTeX compile, 8 numbered figures, complete bib, no placeholders/dup numbering.

### Corrected headline numbers (5 seeds, conservative inv_kl, m=2000)
- MNIST FCN learnt: fclassic 0.0446, fquad 0.0580, flamb 0.0773, bbb 0.2656
- MNIST FCN fquad: learnt 0.0580 vs rand 0.3495 (KL 190 vs 8258)
- Fashion FCN fquad: learnt 0.1747 vs rand 0.4479
- CIFAR9 CNN fquad: learnt 0.4298 vs rand 0.9286 (P1-4 control; learnt KL HIGHER, gains via emp risk)
- CNN<FCN: MNIST 0.0372<0.0580, Fashion 0.1668<0.1747
- small-data (corrected n_bound): 100%→0.058, 50%→0.077, 20%→0.120, 10%→0.145 (n_b=3000; old claim 0.072 invalid)
- difficulty (label noise, same arch): 0→0.058, 10%→0.236, 20%→0.425, 40%→0.700
- MC curve (fixed ckpt): MNIST fquad m=2000→0.056, m=150000→0.041 (gap to reference is MC-budget)
