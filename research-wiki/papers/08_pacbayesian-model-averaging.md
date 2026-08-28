# PAC-Bayesian model averaging

- **Authors:** David A. McAllester
- **Venue/Year:** COLT '99: Proceedings of the twelfth annual conference on Computational learning theory (ACM), pp. 164-170 (1999)
- **ID:** DOI:10.1145/307400.307435 — ✓ verified
- **Contribution:** Origin of the PAC-Bayes theorem: for a posterior Q over hypotheses and a prior P over bounded-loss classifiers, with probability >= 1-delta the Gibbs risk of Q is bounded by its empirical risk plus a KL(Q||P)+log(2sqrt(n)/delta) complexity term, all under a sqrt. Introduces the stochastic / model-averaging predictor whose risk is the expectation over Q rather than a single draw.
- **Relevance to our RQ:** Seed paper for the entire PAC-Bayes lineage. Defines the Gibbs/randomised predictor our 'randomised' predictor rule evaluates, and supplies the fclassic KL+sqrt bound form that fquad and flambda tighten (H1). The McAllester bound is the baseline every objective in Pérez-Ortiz 2021 is compared against.
