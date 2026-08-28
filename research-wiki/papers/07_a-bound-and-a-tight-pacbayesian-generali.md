# A Bound and a Tight PAC-Bayesian Generalization Bound (Thiemann et al. -- source of flambda)

- **Authors:** Niklas Thiemann, Christian Igel, Olivier Wintenberger, Yevgeny Seldin
- **Venue/Year:** ALT 2017 (Algorithmic Learning Theory), PMLR v76; arXiv:1608.05610 (2017)
- **ID:** arXiv:1608.05610 — ✓ verified
- **Contribution:** Derives the PAC-Bayes-lambda (ternary/quasiconvex) bound that Pérez-Ortiz 2021 adopt as their flambda objective (Eq. 12). The bound is a relaxation of the PAC-Bayes-kl bound that is convex (strongly quasiconvex) in both posterior and prior, optimized over a parameter lambda, and is the source of the alternating-minimization procedure (over Q and lambda) used in PBB for flambda.
- **Relevance to our RQ:** flambda is one of the three core objectives in our study (H1) and the strongest in test-error terms on MNIST per Table 1. CORRECTION: the input library listed arXiv:1702.08649 which is WRONG (resolves to an optimization paper); the correct ID is 1608.05610, verified via the arXiv abstract page and PMLR proceedings (thiemann17a). Essential for correctly implementing/reproducing flambda.
