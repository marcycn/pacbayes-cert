# Third-party provenance and modifications

## Upstream
* Repository: PAC-Bayes with Backprop (PBB), accompanying
  Pérez-Ortiz, Rivasplata, Shawe-Taylor & Szepesvári, *"Tighter risk certificates
  for neural networks"*, JMLR 2021 (arXiv:2007.12911).
* Vendored copy: `third_party/PBB/` (probabilistic layers, objectives, inv_kl).
* License: see upstream repository. Retained unmodified under `third_party/` for
  audit; the clean pipeline lives in `src/pacbayes_cert/` and does not import it.

## What the clean reimplementation changes vs PBB (`src/pacbayes_cert/`)
| Area | PBB behaviour | Corrected behaviour | Tag |
|---|---|---|---|
| Bound sample size | `n_bound = len(loader.dataset)` (full dataset) | `n_bound = |S\S0|` from explicit indices | P0-1 |
| Split semantics | implied 3-way disjoint in paper | S0 ⊂ S, bound = S\S0, posterior on all S | P0-2 |
| Certificate scope | posterior-mean "automatically" covered | Gibbs-only; mean/ensemble descriptive | P0-3 |
| Confidence | single `delta_test` reused, called 99% | `delta_mc + delta_pac`, union bound | P0-5 |
| MC aggregation | `/= batch_id` (off-by-one), unweighted | example-weighted, batching-invariant | P0-6 |
| Seeds | global reset to 7/0/10 | decomposed split/model/loader/mc seeds | P0-7 |
| Small-data subset | prefix `range(n)` then shuffle | uniform random permutation, never reads labels | P1-9 |

The probabilistic math (Gaussian/Laplace KL, sampling, truncated-normal init, the
four training objectives, the binary-KL inversion) is reproduced faithfully; only
a latent undefined-variable bug in `ProbConv2d` zeros/random prior init was fixed.

## Legacy results
`legacy_invalidated/` holds the original pre-fix outputs, retained for audit only.
They must not appear in the abstract, tables, figures, conclusion or video.
