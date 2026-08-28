# Speaker script — MSc project video (~6–8 min)

*Keyed to `talk.pdf` (11 frames). Target ~140 wpm → ~7 min. Pause at each [slide].*

**[1 — Title]** Hi. My MSc project is *Certifying Deep Learning Classifier Predictions with High Confidence* — a study of PAC-Bayes-with-Backprop across three image datasets of increasing difficulty.

**[2 — The problem]** Deep networks generalise remarkably well, despite having far more parameters than training data. But the usual evaluation — report error on a held-out test set — gives no *formal* guarantee about how the model will behave on truly unseen data. What I'd like instead is a **risk certificate**: a bound on the true error that holds with a chosen confidence, say 99%. PAC-Bayes theory can provide exactly that — but historically these bounds were *vacuous*, larger than 1, for deep networks.

**[3 — PAC-Bayes]** The idea is to make the predictor *stochastic*: instead of a single weight vector, learn a *distribution* Q over weights and predict by sampling. PAC-Bayes then bounds the true risk of this stochastic predictor by its empirical risk plus a complexity penalty proportional to the KL divergence between the learned posterior Q and a prior P. The bound is tight when Q is close to P. Two things make it non-vacuous: randomise the predictor so the KL is finite, and choose a prior close to the posterior.

**[4 — PBB]** PAC-Bayes-with-Backprop, from Pérez-Ortiz and colleagues in 2021, takes the natural next step: train the posterior Q by *directly minimising a PAC-Bayes bound as the loss*. I compare three bound-derived objectives — fquad and flambda, which are tight relaxations, against the looser classic bound — plus Bayes-by-Backprop, which optimises an ELBO and is *not* a generalisation bound. I also vary the prior — data-free versus data-dependent, learned on a subset of the data — and report three predictor rules.

**[5 — Setup]** I run this on a difficulty ladder: MNIST, where I reproduce the reference paper's Table 1; Fashion-MNIST, which is my extension; and CIFAR-10, for analysis. Fully connected and convolutional networks, trained with SGD, 100 epochs, certificates computed by PAC-Bayes-kl inversion. That's 26 runs in total, every number traceable to a versioned experiment.

**[6 — MNIST reproduction]** The reproduction is faithful: my test errors match the published ones to within four-tenths of a percentage point. The key pattern is clear — Bayes-by-Backprop gives the best accuracy but the loosest certificate, because it isn't minimising a bound; and switching from a data-free to a learned prior tightens the fquad certificate about eight-fold, from 0.33 down to 0.039.

**[7 — Fashion-MNIST]** The methodology transfers to Fashion-MNIST. Every qualitative finding reproduces, and the learned prior again tightens the certificate — here about three-fold, from 0.43 to 0.15. The figure shows this is the dominant lever: the prior choice matters far more than the objective choice.

**[8 — Cross-dataset]** Reading across the ladder, the learnt-prior fquad certificate rises monotonically with difficulty — 0.039 on MNIST, 0.145 on Fashion-MNIST, 0.410 on CIFAR-10. The bound stays informative on the easier datasets and only approaches vacuity on CIFAR-10, where the irreducible error itself is large. This is exactly what PAC-Bayes theory predicts, since the certificate upper-bounds the true risk.

**[9 — Small-data]** A small-data ablation shows the self-certified bound degrades gracefully: even trained on just 10% of the data, the certificate only roughly doubles, staying non-vacuous at 0.072. That's the property that makes PAC-Bayes attractive in low-data regimes.

**[10 — Takeaways]** Three takeaways. First, bound-minimising training works — it gives both good predictors and quantifiable guarantees. Second, the prior is everything: data-dependent priors are the single largest lever on certificate tightness. Third, the most interesting open question is whether we can certify the predictor we *actually* deploy — the deterministic posterior mean — rather than only the stochastic one the bound formally controls.

**[11 — Thanks]** Thank you. The full report, code, and all 26 experiments are in the project repository, and the reference reproduced is Pérez-Ortiz et al., JMLR 2021. I'm happy to take questions.
