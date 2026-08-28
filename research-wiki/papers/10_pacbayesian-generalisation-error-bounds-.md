# PAC-Bayesian Generalisation Error Bounds for Gaussian Process Classification

- **Authors:** Matthias Seeger
- **Venue/Year:** Journal of Machine Learning Research, 3:233-269 (2002)
- **ID:** DOI:10.1162/153244303765208386 — ✓ verified
- **Contribution:** Independently re-derives McAllester's PAC-Bayes bound and specialises it to Gaussian Process classification, claiming the tightest distribution-free generalisation error bounds for approximate Bayesian GPC at the time. Isolates the seeger/seeger-kl form relating the Gibbs risk to a convex function of the empirical risk and KL(Q||P).
- **Relevance to our RQ:** Co-founding reference (with McAllester) for the bound family. The 'Seeger bound' is one of the convex/kl-relaxation objectives the PBB objectives generalise; understanding it is needed to place fquad/flambda/fclassic in the same family tree. Confirms the bound form is shared across GP and neural PBB settings.
