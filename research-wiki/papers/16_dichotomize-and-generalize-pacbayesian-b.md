# Dichotomize and Generalize: PAC-Bayesian Binary Activated Deep Neural Networks

- **Authors:** Gaël Letarte, Pascal Germain, Benjamin Guedj, François Laviolette
- **Venue/Year:** Advances in Neural Information Processing Systems (NeurIPS) 32 (arXiv:1905.10259) (2019)
- **ID:** arXiv:1905.10259 — ✓ verified
- **Contribution:** Trains deep nets with binary (sign) activations by directly optimising a PAC-Bayes bound, parameterising each layer so the network output is a weighted majority vote of binary functions (PBGNet), enabling tractable PAC-Bayes generalization bounds using a data-dependent prior learnt on a subset. Reports non-vacuous, competitive risk certificates on MNIST via a different parametrisation than Gaussian-mean-field PBB.
- **Relevance to our RQ:** Prior-art baseline for data-dependent priors in deep PAC-Bayes: the 'transfer subset' prior-learning recipe predates Pérez-Ortiz 2021 here. Reinforces that the learnt/subset-prior direction is well-established -- another datapoint that H4 is reproduction territory rather than discovery. Useful as the original source of the subset-prior recipe we may re-implement, and as a cross-family comparison point for how prior/parametrisation affects certificate tightness (H4). ID verified via arXiv abstract page.
