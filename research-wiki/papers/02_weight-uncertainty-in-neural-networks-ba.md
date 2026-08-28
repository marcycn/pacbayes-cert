# Weight Uncertainty in Neural Networks (Bayes by Backprop)

- **Authors:** Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra
- **Venue/Year:** ICML 2015; PMLR Vol. 37, pp. 1613-1622 (arXiv:1505.05424) (2015)
- **ID:** arXiv:1505.05424 — ✓ verified
- **Contribution:** Introduces 'Bayes by Backprop' (BBB): reparameterize a Gaussian variational posterior over weights and optimise the expected-NLL-plus-KL ELBO via standard backprop, enabling scalable learning of weight distributions. This is the fbbb objective used as a training-objective baseline in PBB. Uses the reparameterization trick to make the expectation differentiable so gradients flow through sampled weights. Shows weight uncertainty improves robustness/regularization on toy and MNIST tasks.
- **Relevance to our RQ:** Foundational to the entire PBB lineage: BBB is what makes learning weight distributions via backprop practical, and fbbb is one of the four training objectives benchmarked in Pérez-Ortiz 2021 (fbbb gets better test error but much looser risk certificates than fquad/flambda because it does not directly minimize a PAC-Bayes bound). BBB-ELBO is the natural objective ablation baseline to show PAC-Bayes objectives give tighter certificates at comparable accuracy. Named as a foundational item in the charter; reproduced in the official PBB repo.
