# PAC-Bayesian Supervised Classification: The Thermodynamics of Statistical Learning

- **Authors:** Olivier Catoni
- **Venue/Year:** IMS Lecture Notes-Monograph Series, Vol. 56 (Institute of Mathematical Statistics); arXiv:0712.0248 (2007)
- **ID:** arXiv:0712.0248 — ✓ verified
- **Contribution:** Derives two 'tempered'/localised PAC-Bayes oracle bounds for the Gibbs posterior Q_lambda proportional to exp(-lambda*Rhat)P, for binary 0-1 loss: (i) a sqrt-rate bound where the gap is ~sqrt((KL+ln(1/delta))/n), and (ii) a faster, log-rate bound where the gap scales as (KL+ln(1/delta))/lambda + lambda/n, optimised over the inverse temperature lambda. These are tighter than the classical McAllester bound when empirical risk is small.
- **Relevance to our RQ:** Direct theoretical source of H1. flambda corresponds to optimising Catoni's inverse-temperature lambda bound, and fquad (quadratic tempering) to the refined second-order/log-rate variant. Catoni is the reason the tempered objectives can beat fclassic at low empirical risk. The data-dependent/localised-prior ideas also prefigure H4. ID verified via arXiv abstract page.
