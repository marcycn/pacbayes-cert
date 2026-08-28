# A Simple Baseline for Bayesian Uncertainty in Deep Learning (SWAG)

- **Authors:** Wesley J. Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, Andrew Gordon Wilson
- **Venue/Year:** NeurIPS 2019 (Advances in Neural Information Processing Systems 32; arXiv:1902.02476) (2019)
- **ID:** arXiv:1902.02476 — ✓ verified
- **Contribution:** Approximates the posterior over weights with a low-rank-plus-diagonal Gaussian fit to the SGD trajectory tail (Stochastic Weight Averaging), giving well-calibrated uncertainty and improving test error at low cost; scalable to large CNNs/ResNets.
- **Relevance to our RQ:** Strong modern Bayesian-DL baseline directly competing with PBB on accuracy+uncertainty. Unlike PBB, SWAG has no formal risk certificate. Must compare on CIFAR-10 especially (H3); useful as a learned-prior / posterior-init source for our data-dependent priors (H4).
