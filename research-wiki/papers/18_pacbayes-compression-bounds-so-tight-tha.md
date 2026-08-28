# PAC-Bayes Compression Bounds So Tight That They Can Explain Generalization

- **Authors:** Sanae Lotfi, Marc Finzi, Sanyam Kapoor, Andreas Wormuth, Soeren Laue, Gintare Karolina Dziugaite, Alex Wilson, Daniel Roy
- **Venue/Year:** NeurIPS 2022 (Advances in Neural Information Processing Systems 35; arXiv:2211.13609) (2022)
- **ID:** arXiv:2211.13609 — ✓ verified
- **Contribution:** Develops PAC-Bayes compression bounds by quantizing network parameters in a learned linear subspace, yielding numerically strong non-vacuous generalization bounds for realistic deep architectures (ResNets, CNNs) on CIFAR-10/100 and ImageNet; substantially tighter than Zhou et al. (2019).
- **Relevance to our RQ:** Major 2022 SOTA comparison point for non-vacuous bounds on CIFAR-10 -- the dataset where our H3 predicts cert tightness degrades. Although a compression (not weight-distribution) route, it defines the current non-vacuous-bound frontier our PBB certificates should be benchmarked against, and its CIFAR-10 numbers anchor H3. ID verified via arXiv abstract page.
