# Evidential Deep Learning to Quantify Classification Uncertainty

- **Authors:** Murat Sensoy, Lance Kaplan, Melih Kandemir
- **Venue/Year:** NeurIPS 2018 (Advances in Neural Information Processing Systems 31, pp. 3183-3193; arXiv:1806.01768) (2018)
- **ID:** arXiv:1806.01768 — ✓ verified
- **Contribution:** Replaces softmax with a Dirichlet output placed via evidence/Dempster-Shafer theory, learning a higher-order distribution over class-probability distributions with a single forward pass (no sampling) for both aleatoric and epistemic uncertainty.
- **Relevance to our RQ:** Deterministic, single-pass uncertainty baseline to contrast with our sampling-based randomised predictor. Has no frequentist risk guarantee; positions PBB certificates as a more rigorous alternative for high-confidence classification.
