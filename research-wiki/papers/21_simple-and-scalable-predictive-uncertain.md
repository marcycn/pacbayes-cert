# Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles

- **Authors:** Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell
- **Venue/Year:** NeurIPS 2017 (Advances in Neural Information Processing Systems 30; arXiv:1612.01474) (2017)
- **ID:** arXiv:1612.01474 — ✓ verified
- **Contribution:** Shows that training an ensemble of M independently-initialized probabilistic nets (with NLL/proper-scoring-rule loss + adversarial training) yields better-calibrated, more accurate uncertainty than Bayesian NN approaches of the era; introduces the standard 'deep ensembles' baseline for OOD detection and predictive variance.
- **Relevance to our RQ:** Primary non-Bayesian uncertainty baseline we must beat/discuss. Ensembles give competitive test error but NO distributional risk certificate, only heuristic confidence -- directly motivates our PAC-Bayes value-add (certified high-confidence predictions, RQ/H3). Relevant to MC-ensemble predictor-rule comparison.
