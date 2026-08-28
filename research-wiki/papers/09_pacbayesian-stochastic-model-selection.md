# PAC-Bayesian Stochastic Model Selection

- **Authors:** David A. McAllester
- **Venue/Year:** Machine Learning, 51(1):5-21 (Kluwer/Springer) (2003)
- **ID:** DOI:10.1023/A:1021840411064 — ✓ verified
- **Contribution:** Refines the original bound into the now-standard journal form: with prob >= 1-delta simultaneously for all Q, E_{h~Q} L(h) <= E_{h~Q} Lhat(h) + sqrt( (KL(Q||P) + ln(2sqrt(n)/delta)) / n ). Establishes that the bound applies to stochastic model selection and that the KL term is the data-free complexity penalty. DOI verified via Springer link.springer.com/article/10.1023/A:1021840411064.
- **Relevance to our RQ:** The canonical statement of the 'classical' PAC-Bayes bound. This is exactly the fclassic objective term in Pérez-Ortiz 2021. Cite to explain the fclassic certificate value = empirical risk + sqrt(KL/n)-style term, and as the reference point H1 claims fquad/flambda beat.
