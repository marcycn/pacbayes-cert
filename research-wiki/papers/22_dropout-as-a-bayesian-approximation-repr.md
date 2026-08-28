# Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning

- **Authors:** Yarin Gal, Zoubin Ghahramani
- **Venue/Year:** ICML 2016 (PMLR vol. 48, pp. 1050-1059; arXiv:1506.02142) (2016)
- **ID:** arXiv:1506.02142 — ✓ verified
- **Contribution:** Proves dropout training is equivalent to variational approximate Bayesian inference in a deep Gaussian process; introduces Monte Carlo dropout (run T stochastic forward passes at test time) as a cheap posterior-approximation for epistemic uncertainty.
- **Relevance to our RQ:** Must-discuss alternative to learning explicit weight distributions (PBB/BBB). MC-dropout is the obvious cheap baseline for our 'randomised / MC-ensemble predictor' rule and for calibration comparisons; we position PBB as a more principled posterior giving provable certificates rather than ad-hoc variance.
