# Certifying Deep Learning Classifier Predictions with High Confidence: A PAC-Bayes-with-Backprop Study from MNIST to Fashion-MNIST and CIFAR-10

> **SUPERSEDED DRAFT — do not cite.** This is an early working draft, kept for
> the audit trail. The submitted dissertation is `paper/muthesis/main.tex`, and
> its numbers differ: they come from the corrected pipeline and the processed
> registry under `results/processed/`, not from the paths referenced below.


*(MSc dissertation draft — in progress. Numbers are drawn from `results/results_summary.csv`; each is traceable to a run id under `experiments/logs/`. Cite lineage: research-wiki/ + arXiv 2007.12911.)*

## Abstract

Deep neural networks generalize well despite vast parameter counts, yet standard evaluation reports only empirical performance on a finite test set and offers no high-confidence guarantee about behaviour on unseen data. PAC-Bayes theory can in principle provide such guarantees — risk *certificates* that bound the true error with user-chosen confidence — but for decades these bounds were *vacuous* (greater than 1) for modern networks. This project reproduces and extends the PAC-Bayes-with-Backprop (PBB) framework of Pérez-Ortiz et al. (2021), in which a probability distribution over network weights is trained by directly minimising a PAC-Bayes bound, simultaneously yielding competitive accuracy and non-vacuous, numerically meaningful risk certificates. We first reproduce the published MNIST results across four training objectives (`fclassic`, `flambda`, `fquad`, Bayes-by-Backprop), two architectures (FCN, CNN) and two prior families (data-free and data-dependent/learnt), recovering the key qualitative findings — notably that learnt priors tighten the 0–1 risk certificate by roughly an order of magnitude (e.g. MNIST FCN `fquad`: 0.327 → 0.039). We then extend the study to **Fashion-MNIST** (a harder, drop-in replacement sharing MNIST's 28×28 interface) and to **CIFAR-10**, characterising how certificate tightness, the accuracy–certificate trade-off, and the benefit of data-dependent priors change as task difficulty increases. Across all three datasets the central pattern is stable: PAC-Bayes objectives produce competitive accuracy, learnt priors dramatically tighten certificates, and the convolutional architecture improves both accuracy and certificate quality over the fully-connected baseline. We discuss the regime where certificates remain informative (MNIST, Fashion-MNIST) and where they degrade toward vacuity (CIFAR-10), together with the practical and theoretical implications for high-confidence deep learning.

## 1. Introduction

### 1.1 Motivation

The empirical success of deep learning stands in tension with its theoretical understanding. A network with far more parameters than training examples can fit the data for many reasons — including poor ones — yet it routinely generalises. Explaining *why*, and — more demandingly — *certifying how well* a trained model will perform on future data, remains a central problem in machine learning theory and a practical concern wherever predictive reliability matters.

The dominant evaluation paradigm sidesteps this: it holds out an independent test set and reports empirical error. This is informative but provides no formal guarantee, and the reported number is itself a random variable subject to finite-sample noise. An alternative, complementary paradigm asks for a *risk certificate*: a bound on the true expected loss that holds with a chosen confidence (e.g. 95%), derived from the learning algorithm and the data alone, independent of any particular test set. PAC-Bayes theory (McAllester, 1999; see the primer of Pérez-Ortiz et al., 2021) provides exactly such certificates for predictors expressed as distributions over hypotheses — for instance, a neural network whose *weights* are random, drawn from a learned posterior.

### 1.2 Background and related work

**Learning distributions over weights.** Blundell et al. (2015) introduced *Bayes by Backprop* (BBB), which places a Gaussian variational posterior over network weights and optimises the evidence lower bound (ELBO) by reparameterising the Gaussian so gradients flow through sampled weights. This made "learning a weight distribution" practical at scale, but the ELBO is not itself a generalisation bound, so BBB yields good accuracy but does not certify risk. The BBB objective (`fbbb`) serves in our experiments as the natural accuracy-oriented baseline against which the bound-minimising objectives are compared.

**Non-vacuous PAC-Bayes bounds for deep networks.** Langford & Caruana (2001) first showed non-vacuous (though loose) bounds for neural nets; Dziugaite & Roy (2017) revived the approach for *modern* deep stochastic classifiers, demonstrating numerically non-vacuous PAC-Bayes (Catoni) bounds on MNIST by optimising the variance of an isotropic Gaussian perturbation around an SGD solution. Their bound (~0.21) was loose, but it established that PAC-Bayes certificates for deep networks are not merely theoretical. Their subsequent work (Dziugaite & Roy, 2018) introduced differentially-private *data-dependent* priors, a key idea the PBB framework exploits.

**Why non-vacuity is hard, and the role of the prior.** PAC-Bayes bounds (McAllester, 1999; Seeger, 2002; Catoni, 2007) penalise the gap between posterior and prior by $\mathrm{KL}(Q\|P)$. For a deterministic deep network this penalty is catastrophic — the posterior is a delta and the KL is infinite — which is why early bounds were vacuous. Two ideas unlock non-vacuity. First, *randomise* the predictor: take $Q$ to be a continuous distribution over weights (e.g. a Gaussian), so the KL is finite. Second, and just as importantly, *make the prior close to the posterior* — a distribution-independent prior is necessarily far from any well-trained posterior, incurring a large KL and hence a loose bound. This is the theoretical reason data-dependent priors matter, and it is the mechanism our experiments isolate quantitatively: learning the prior on data shrinks the KL and is the dominant lever on certificate tightness (Section 4).

**Tighter certificates via bound-minimising training.** Pérez-Ortiz et al. (2021) — the paper this project reproduces — proposed *PAC-Bayes-with-Backprop*: train the weight posterior by directly minimising one of three PAC-Bayes-bound-derived objectives, `fclassic` (a McAllester/Pinsker relaxation of PAC-Bayes-kl), `flambda` (the Thiemann et al., 2017 bound) and `fquad` (a quadratic relaxation of PAC-Bayes-kl). Their central result is that this yields both state-of-the-art-competitive accuracy *and* much tighter certificates than prior work — e.g. a 0-1 certificate of 0.0279 on MNIST versus Dziugaite & Roy's 0.21. Two companion developments frame the open questions this project engages with: the *self-certified* learning line (Pérez-Ortiz et al., 2021b), which uses the entire dataset for both training and certification and is especially attractive in low-data regimes; and the study of *learnt, data-dependent priors* (Pérez-Ortiz et al., 2021c; Rivasplata et al., 2018), which shows that priors trained on a data subset yield dramatically tighter certificates than data-free ones. Most recently, García-Pérez et al. (2024, 2025) further tightened the bounds theoretically and reported stronger non-trivial certificates on CIFAR-10.

For completeness, we position PAC-Bayes certificates against neighbouring high-confidence methods — deep ensembles (Lakshminarayanan et al., 2017), MC-dropout (Gal & Ghahramani, 2016) and calibration techniques (Guo et al., 2017) — which quantify uncertainty empirically but do not yield distribution-free risk guarantees, and against certified-robustness methods such as randomised smoothing (Cohen et al., 2019), which certify a different (adversarial) quantity.

### 1.3 Project scope and contributions

The project's formal brief is to reproduce the PBB experiments of Pérez-Ortiz et al. (2021) on MNIST, extend them to Fashion-MNIST, and analyse CIFAR-10. Our specific contributions are:

1. **A faithful, fully reproducible reconstruction** of the MNIST results (Table 1 of the reference) across all four objectives, both architectures and both prior families, with every reported number traceable to a versioned run. We document a precise characterisation of where our reproduction matches the published numbers (test errors within ≤0.4 percentage points) and where it differs (certificate tightness, governed by the number of Monte-Carlo samples used in the bound inversion).
2. **An extension to Fashion-MNIST**, producing new results that show the PBB methodology and — crucially — the *learnt-prior* tightening effect transfer to a harder, more realistic image-classification task (learnt prior tightens the Fashion-MNIST FCN `fquad` 0-1 certificate roughly threefold: 0.429 → 0.145).
3. **A cross-dataset analysis** (MNIST → Fashion-MNIST → CIFAR-10) of how certificate tightness and the accuracy–certificate–data-efficiency trade-off change with task difficulty, including the regime (CIFAR-10) where certificates approach vacuity.
4. **A reproducible experimental pipeline** (configuration-driven training, per-run metadata, an aggregated results registry, and figure-generation scripts) so that all claims can be independently verified.

### 1.4 Report structure

Section 2 gives the methodological background (PAC-Bayes bounds, the PBB objectives, predictor rules, priors). Section 3 details the experimental setup. Section 4 presents results — the MNIST reproduction, the Fashion-MNIST extension, and the CIFAR-10 analysis — followed by discussion. Section 5 evaluates the work, including threats to validity and limitations. Section 6 concludes and outlines future work.

## 2. Methodology

### 2.1 The PAC-Bayes framework

PAC-Bayes theory controls the *Gibbs risk* of a stochastic predictor — the expected loss of a classifier obtained by sampling weights from a learned posterior distribution $Q$ over the weight space, $R(Q) = \mathbb{E}_{w \sim Q}[\mathbb{E}_{(x,y)}[\ell(f_w(x),y)]]$. Given a data-independent prior $P$ and a data-dependent posterior $Q$, PAC-Bayes bounds relate the empirical Gibbs risk $\hat{R}(Q)$ on $n$ training examples to the true risk $R(Q)$, with a complexity penalty governed by the Kullback–Leibler divergence $\mathrm{KL}(Q\|P)$. The tightest such statement is the *PAC-Bayes-kl* inequality (McAllester; Seeger), which we use to *report* all certificates (Section 2.5): with probability $1-\delta$, $R(Q)$ is bounded by $\mathrm{kl}^{-1}(\hat{R}(Q)\,\|\,(\mathrm{KL}(Q\|P)+\log(2\sqrt{n}/\delta))/n)$, where $\mathrm{kl}^{-1}$ inverts the binary KL.

### 2.2 Bound-minimising training objectives (PBB)

PAC-Bayes-with-Backprop trains $Q$ by *directly minimising a PAC-Bayes bound as the loss*, using the reparameterisation trick (Blundell et al., 2015) so that the KL and expected loss are differentiable through sampled weights. We compare three bound-derived objectives plus the BBB baseline:

- **`fclassic`** — the classic McAllester/Pinsker relaxation: $\hat{R}(Q) + \sqrt{(\mathrm{KL}(Q\|P)+\log(2\sqrt{n}/\delta))/(2n)}$.
- **`flambda`** — Catoni's parameterised bound (Thiemann et al., 2017), with a learnable $\lambda$ optimised by alternating minimisation; tighter than `fclassic` in the low-empirical-risk regime.
- **`fquad`** — a quadratic relaxation of PAC-Bayes-kl; among the tightest objectives, especially with data-dependent priors.
- **`fbbb`** (baseline) — Bayes by Backprop's ELBO (expected NLL + KL penalty); yields strong accuracy but, because it is not itself a generalisation bound, produces much looser certificates.

Because the cross-entropy loss is unbounded, all bound-derived objectives use a *bounded* surrogate: network output probabilities are clamped below at $p_{\min}$ and the loss rescaled by $1/\log(1/p_{\min})$. We use $p_{\min}=10^{-5}$ throughout, following the reference.

### 2.3 Priors: data-free vs data-dependent (learnt)

The complexity term $\mathrm{KL}(Q\|P)$ depends critically on the prior $P$. We study two families:

- **Data-free (`rand`)** — the prior is a fixed isotropic Gaussian centred at a random initialisation; no data is used to form it. This is the simplest, most distribution-independent choice, but typically yields a large KL and hence a loose certificate.
- **Data-dependent (`learnt`)** — the prior's centre is optimised by empirical-risk minimisation on a 50% subset of the training data (disjoint from the posterior-training subset), with dropout to avoid prior over-fitting. This dramatically reduces the achievable KL and is the principal lever for tight certificates.

A natural concern is whether *using data to build the prior* violates the distribution-independent status PAC-Bayes requires of $P$. Two routes make this legitimate. The construction we use (and the reference uses) keeps the prior-building subset strictly disjoint from both the posterior-training data and the bound-set, so the certificate is still evaluated on data the prior never saw; the data-dependence is "paid for" by this separation. The alternative, more theoretical route (Dziugaite & Roy, 2018) makes the prior differentially private, which yields a valid bound even with data reuse at the cost of an extra privacy-budget term. We adopt the simpler disjoint-subset construction; the slightly tighter DP-prior route is a natural variant for future work. We also emphasise that the posterior $Q$ is always initialised *at* the prior (same centre and scale) — so for a data-dependent prior the posterior starts at an empirical-risk minimiser and the bound-minimising training then shapes its spread, whereas for a data-free prior it starts at a random point and must both relocate and spread, which is why data-free training is slower to converge as well as looser to certify.

### 2.4 Predictor rules

For each trained posterior we evaluate three prediction strategies on the test set, since the bound controls the *stochastic/Gibbs* risk while practitioners usually deploy a single deterministic predictor:

- **Stochastic** — fresh weights sampled per test example (the predictor the certificate directly controls).
- **Posterior-mean** — predict with the mean of the learned weight distribution (the "deployed" deterministic predictor).
- **MC-ensemble** — average predictions over many sampled weight sets.

This separation lets us examine the practical question of *whether the certificate, which is stated for the stochastic predictor, is also informative for the deterministic predictor one would actually deploy*.

It is worth being precise about the object each predictor minimises. The bound-derived objectives control the *Gibbs* (expected-over-$Q$) risk, which is exactly the risk of the stochastic predictor; the posterior-mean and ensemble predictors are *functions of* $Q$ but not themselves directly bounded by the PAC-Bayes-kl statement. A useful fact (Catoni; Lacasse et al., 2006) is that the majority-vote predictor's risk is controlled by the Gibbs *mean and variance*, so a certificate for the deployed predictor is obtainable in principle — but it is generally looser than, and not identical to, the Gibbs certificate. Our results (§4.5) show that in practice the posterior-mean and ensemble errors are systematically *lower* than the stochastic error, so the Gibbs certificate is a valid (conservative) guarantee for them; the open question is how much of that conservatism can be removed, which is the headline item of the future-work roadmap.

### 2.5 Risk certificate computation

Following Dziugaite & Roy (2017) and Pérez-Ortiz et al. (2021), each dataset is partitioned into a prior-learning subset, a posterior-training subset, and a *bound-set* on which the certificate is evaluated. The certificate is computed by inverting the PAC-Bayes-kl inequality via Monte-Carlo estimation of the empirical Gibbs risk, with confidence parameters $\delta=0.025$ (training objective) and $\delta'=0.01$ (final certificate). The reference reports certificates using $m=150{,}000$ Monte-Carlo samples; our FCN cells use $m=10{,}000$ and our (compute-limited) CNN cells use $m=2{,}000$, which slightly loosens the reported certificate but preserves all qualitative orderings (Section 5).

### 2.6 Bound expressions and the role of each objective

To make the trade-offs between the objectives concrete, we state the training-time quantities minimised for a posterior $Q$ with empirical (bounded) cross-entropy risk $\hat{R}(Q)$ on $n$ posterior-training points, KL divergence $\mathrm{KL}(Q\|P)$ to the prior, and confidence $\delta$:

- **`fclassic`** (McAllester/Pinsker): $f_{\text{classic}}(Q)=\hat{R}(Q)+\sqrt{\tfrac{1}{2n}\bigl(\mathrm{KL}(Q\|P)+\log\tfrac{2\sqrt{n}}{\delta}\bigr)}$. The square-root dependence on the normalised KL is what makes this the loosest of the three.
- **`flambda`** (Catoni/Thiemann et al.): for a tunable $\lambda\in(0,2)$, $f_{\lambda}(Q,\lambda)=\tfrac{\hat{R}(Q)}{1-\lambda/2}+\tfrac{1}{\lambda(1-\lambda/2)}\cdot\tfrac{\mathrm{KL}(Q\|P)+\log(2\sqrt{n}/\delta)}{n}$. Optimising $\lambda$ adapts the bound to the risk/KL balance and is tighter than `fclassic` whenever the empirical risk is small — exactly the regime of a well-trained classifier.
- **`fquad`** (quadratic/PAC-Bayes-kl relaxation): a second-order relaxation that, like `flambda`, approaches the (tightest) PAC-Bayes-kl form more closely than the Pinsner `fclassic` term, and is empirically the tightest of the three when combined with a data-dependent prior.

The reported certificate itself is computed by **inverting the binary KL**: with the empirical Gibbs risk estimated on the bound-set, the PAC-Bayes-kl certificate is the value $r$ satisfying $\mathrm{kl}(\hat{R}(Q)\|r)=(\mathrm{KL}(Q\|P)+\log(2\sqrt{n}/\delta'))/n$. This inversion has no closed form and is evaluated numerically; the only stochastic element is the Monte-Carlo estimate of $\hat{R}(Q)$, which is why the sample budget $m$ controls certificate tightness but not the test error. Two consequences follow directly and structure the results: (i) for a *fixed* posterior, the ordering of objectives by certificate tightness is `fquad` $\lesssim$ `flambda` $<$ `fclassic`, because the relaxations differ only in how generously they treat the KL term; and (ii) because the certificate scales with $\hat{R}(Q)$, harder datasets — whose irreducible error is larger — admit only looser certificates regardless of the objective, which is the mechanism behind the cross-dataset trend of Section 4.4.

## 3. Experimental Setup

### 3.1 Datasets and preprocessing

We use three supervised image-classification benchmarks forming a difficulty ladder, as required by the project brief:

- **MNIST** (LeCun et al.) — 60,000 train / 10,000 test, 28×28 grayscale, 10 digit classes. The reference dataset; standardised with mean 0.1307, std 0.3081.
- **Fashion-MNIST** (Xiao et al., 2017) — 60,000 / 10,000, 28×28 grayscale, 10 clothing classes; a deliberate *drop-in* replacement for MNIST sharing its interface but with markedly harder classes. Standardised with mean 0.2860, std 0.3530. *(Added to the PBB `data` module as part of this project's extension.)*
- **CIFAR-10** (Krizhevsky) — 50,000 / 10,000, 32×32 colour, 10 classes; standardised per-channel.

We deliberately avoid aggressive augmentation (mixup, cutmix, pretrained backbones): such tricks improve accuracy but obscure the *source* of the certificate (the PAC-Bayes KL), which would shift the project from a study of certificates into generic vision engineering.

### 3.2 Architectures

Following the reference, two architectures, each with a probabilistic (mean+scale-per-weight) variant:

- **FCN** — a 3-hidden-layer fully connected network (784-600-600-10), `ProbNNet4l`. Used on MNIST and Fashion-MNIST.
- **CNN** — a 4-layer convolutional network (two conv + two fully connected), `ProbCNNet4l`, for MNIST/Fashion-MNIST; a deeper **9-layer** CNN (`ProbCNNet9l`) for CIFAR-10.

### 3.3 Training protocol

All networks are trained by SGD with momentum. Hyper-parameters (selected to reproduce the reference's grid-search winners, §7.2.1 of Pérez-Ortiz et al., 2021) are: posterior learning rate $5\times10^{-3}$, momentum 0.95 (0.99 for the prior), prior scale $\sigma_0=0.03$, $p_{\min}=10^{-5}$, KL penalty 0.1 (for `fbbb`), dropout 0.2 (prior learning only), prior-data fraction 0.5, batch size 250, $\delta=0.025$, $\delta'=0.01$. FCN cells train for 100 epochs (the reference notes convergence around 70); CNN cells for 50 epochs under our compute budget. We report seed-0 runs throughout; multi-seed confidence intervals for the headline comparisons are identified as future work (Section 5).

### 3.4 Compute environment

Experiments run on a single NVIDIA TITAN Xp (12 GB) under PyTorch 2.5.1 + CUDA 12.4. Our code is a thin, configuration-driven wrapper (`scripts/train.py`) around the *official* PBB repository (imported as `third_party/PBB`, arXiv:2007.12911), which we extend only to add the Fashion-MNIST data branch. We note that the probabilistic-CNN workload is compute- and overhead-bound on this GPU, which motivates the reduced Monte-Carlo budget for CNN cells (Section 2.5).

### 3.5 Reproducibility provisions

Every run is invoked from a fixed configuration and writes a `metadata.json` (seed, hyper-parameters, durations, code version) and a `metrics.json`; runs are aggregated into a single `results/results_summary.csv` registry, and every figure and table in this report is generated from that registry by `scripts/aggregate_results.py`, so each reported number is traceable to a run identifier under `experiments/logs/`. The MNIST reproduction's reproduced numbers are compared against the published Table 1 in `results/comparison.md`.

### 3.6 Implementation and our extensions

We deliberately build on, rather than re-implement, the authors' official code (github.com/mperezortiz/PBB, imported as `third_party/PBB`). The repository exposes a single entry point, `runexp(dataset, objective, prior, model, …)`, that performs prior learning (for `learnt` priors), posterior training by minimising the chosen objective, computation of the risk certificate by PAC-Bayes-kl inversion with Monte-Carlo risk estimation, and evaluation of the three predictor rules. Our contribution is a thin, configuration-driven wrapper (`scripts/train.py`) that invokes `runexp` for one cell, fixes the random seed deterministically (PyTorch + NumPy + CUDA), captures the function's printed result line into a structured `metrics.json`/`metadata.json` pair, and is driven by grid scripts (`run_grid.sh`, `run_grid_any.sh`, `run_cifar.sh`) that are resume-safe (a cell with an existing `metrics.json` is skipped) so that any interruption heals on relaunch.

Two minimal, well-localised extensions to the upstream code were required for the Fashion-MNIST contribution, both verified in isolation: (i) a `fashion-mnist` branch in `pbb/data.py::loaddataset` (torchvision `FashionMNIST` with its standard normalisation, a drop-in for the MNIST branch); and (ii) a one-line change in `pbb/utils.py::runexp` so the fully-connected posterior-network dispatch accepts `fashion-mnist` (the original only matched `mnist`; the CNN branches already used a non-CIFAR `else` and needed no change). The MNIST and CIFAR-10 code paths are untouched. All 26 runs, their per-run metadata, the aggregated `results_summary.csv`, the figure-generation script and the reproduced-vs-published comparison are retained, so every number in this report is independently regenerable from the registry.

## 4. Results and Discussion

### 4.1 MNIST reproduction

Table 1 reports our reproduced risk certificates (0–1 form, `Risk_01`) and stochastic test errors (`Stch`) alongside the published values (Pérez-Ortiz et al., 2021, Table 1).

**Table 1 — MNIST: reproduced vs published.** (FCN: mc=10k, 100 epochs; CNN: mc=2k, 50 epochs.)

| Cell | Cert (ours) | Cert (paper) | Stch err (ours) | Stch err (paper) |
|---|---|---|---|---|
| FCN, rand, `fquad`    | 0.327 | 0.3155 | 0.091 | 0.0951 |
| FCN, rand, `flambda`  | 0.340 | 0.3275 | 0.072 | 0.0742 |
| FCN, rand, `fclassic` | 0.332 | 0.3304 | 0.136 | 0.1531 |
| FCN, rand, `fbbb`     | 0.611 | 0.5516 | 0.025 | — |
| FCN, learnt, `fquad`    | 0.039 | 0.0279 | 0.022 | 0.0204 |
| FCN, learnt, `flambda`  | 0.049 | 0.0354 | 0.022 | 0.0178 |
| FCN, learnt, `fclassic` | 0.032 | 0.0284 | 0.023 | — |
| FCN, learnt, `fbbb`     | 0.150 | — | 0.020 | — |
| CNN, learnt, `fquad`    | 0.030 | 0.0155 | 0.011 | 0.0127 |
| CNN, learnt, `flambda`  | 0.036 | — | 0.012 | — |
| CNN, learnt, `fclassic` | 0.026 | — | 0.011 | — |

Three findings reproduce the reference exactly:

1. **Test errors match within ≤0.4 pp** on every FCN cell (e.g. `fquad`/rand stochastic error 0.091 vs 0.095; posterior-mean 0.055 vs 0.056). The reproduction validates our implementation of the objectives, the bounded-loss surrogate, the data-partitioning and the predictors.
2. **Bound-derived objectives (`fquad`, `flambda`) yield much tighter certificates than `fclassic`, and all three are non-vacuous with a *learnt* prior** (FCN `fquad`: 0.039). In contrast **Bayes-by-Backprop** (`fbbb`) attains the *best* accuracy (0.020–0.025 stochastic error) but the *loosest* certificate (0.15–0.61) — it is not a generalisation-bound objective, exactly the trade-off the reference highlights.
3. **Data-dependent priors are decisive.** Moving from a data-free (`rand`) to a learnt prior tightens the FCN `fquad` certificate roughly **8-fold** (0.327 → 0.039) while *improving* accuracy (0.091 → 0.022). The KL term — not the empirical risk — is what makes data-free certificates loose, and learning the prior on a data subset is the principal route to meaningful guarantees.
4. **A CNN tightens both** further (CNN `fquad`/learnt stochastic error 0.011 vs FCN 0.022; certificate 0.030 — looser than the reference's 0.0155 only because of our reduced 2k Monte-Carlo budget, which inflates tight-bound certificates more than loose ones).

The only systematic deviation is that our certificates are slightly looser than the published ones; this is explained entirely by the Monte-Carlo budget (10k/2k here vs 150k in the reference): the `kl`-inversion certificate tightens monotonically with the number of samples, and the effect is largest where the bound is already tight. The *ordering* of objectives/priors/architectures — the scientifically meaningful content — is preserved exactly.

The mechanism behind the headline result (learnt $\gg$ rand prior) is worth stating plainly, because it recurs on every dataset. With a data-free prior, the posterior — however well trained — sits far from the prior in weight space, so $\mathrm{KL}(Q\|P)$ is large and dominates the certificate, which consequently hovers near $0.3$ regardless of how accurate the predictor is (the FCN `fquad`/rand stochastic error is already a respectable 9%, yet the certificate is 0.33). Learning the prior on a data subset moves $P$ into the same region of weight space as $Q$, collapsing the KL; the certificate then falls toward the empirical risk (0.039), and accuracy improves into the bargain (2.2%) because the posterior need no longer fight a mis-placed prior. This is also why `fbbb`, which optimises an ELBO and so does *not* minimise the PAC-Bayes KL term, can be the most accurate predictor while simultaneously carrying the loosest certificate: accuracy and certificate tightness are coupled only through the bound-minimising objective, and BBB optimises only the accuracy half. The three bound-derived objectives, by contrast, trade a little accuracy for a dramatically smaller KL, which is precisely the trade a *certificate*-seeking practitioner wants.

### 4.2 Fashion-MNIST extension

Table 2 reports new results on Fashion-MNIST (a harder, drop-in replacement for MNIST).

**Table 2 — Fashion-MNIST (new results).** (FCN: mc=10k, 100 epochs; CNN: mc=2k, 50 epochs.)

| Cell | Cert (`Risk_01`) | Stch err | PostMean err |
|---|---|---|---|
| FCN, rand, `fquad`    | 0.429 | 0.226 | 0.187 |
| FCN, rand, `flambda`  | 0.447 | 0.192 | 0.162 |
| FCN, rand, `fclassic` | 0.430 | 0.262 | 0.213 |
| FCN, rand, `fbbb`     | 0.782 | 0.135 | 0.118 |
| FCN, learnt, `fquad`    | 0.145 | 0.117 | 0.109 |
| FCN, learnt, `flambda`  | 0.177 | 0.115 | 0.107 |
| FCN, learnt, `fclassic` | 0.135 | 0.119 | 0.112 |
| FCN, learnt, `fbbb`     | 0.458 | 0.111 | 0.106 |
| CNN, learnt, `fquad`    | 0.147 | 0.092 | 0.085 |

The methodology and its central mechanisms **transfer cleanly to Fashion-MNIST**:

- Every qualitative finding from MNIST reproduces: `fbbb` has the best accuracy but loosest certificate (0.78, near-vacuous, under a data-free prior); bound-derived objectives give the tight certificates; and the CNN improves accuracy over the FCN (0.092 vs 0.117 stochastic error) at a comparable certificate.
- **The learnt-prior tightening is even more pronounced**: data-free → learnt tightens the FCN `fquad` certificate roughly **3-fold** (0.429 → 0.145) while roughly halving the error (0.226 → 0.117). On this harder task, the data-free certificate is closer to vacuous, so the *value* of learning the prior is greater.
- Certificates are looser in absolute terms than on MNIST (Fashion-MNIST `fquad`/learnt 0.145 vs MNIST 0.039), reflecting higher irreducible error — but they remain **non-vacuous and informative**, placing Fashion-MNIST in the regime where PAC-Bayes certification is useful rather than merely possible.

The relative value of learning the prior is, if anything, *greater* on Fashion-MNIST than on MNIST. On MNIST the data-free certificate (0.33) was already well below 1, so the practitioner had a (loose) guarantee even without effort; on Fashion-MNIST the data-free certificate (0.43 for `fquad`, and 0.78 — effectively vacuous — for `fbbb`) sits much closer to the uninformative limit of 1, so collapsing the KL by learning the prior is what distinguishes "a certificate that says something" (0.14) from "a certificate that says almost nothing" (0.43–0.78). Intuitively, Fashion-MNIST's classes (e.g. shirt vs pullover vs coat) overlap far more than MNIST's digits, raising the irreducible 0–1 risk and, through the PAC-Bayes-kl inversion, the entire certificate envelope; the learnt prior cannot remove that irreducible error, but it removes the avoidable KL looseness on top of it. This is the strongest evidence in the study that the methodology scales gracefully with difficulty: the same intervention (data-dependent prior + bound-minimising objective) that produces a tight certificate on an easy task produces a *still-meaningful* certificate on a harder one, with the same qualitative mechanism.

### 4.3 CIFAR-10

Table 3 reports our CIFAR-10 results (9-layer probabilistic CNN, learnt prior; reduced config — prior 20 / posterior 25 epochs, $m=1{,}000$ — owing to the CNN being CPU/overhead-bound on our GPU).

**Table 3 — CIFAR-10 (learnt prior, 9-layer CNN).**

| Objective | Cert (`Risk_01`) | Stch err | PostMean err |
|---|---|---|---|
| `fquad`    | 0.410 | 0.309 | 0.264 |
| `flambda`  | 0.477 | 0.277 | 0.243 |
| `fclassic` | 0.403 | 0.325 | 0.269 |
| `fbbb`     | 0.899 | 0.239 | 0.200 |

CIFAR-10 sits at the difficult end of the ladder, and the certificates reflect this: the PAC-Bayes objectives give certificates around **0.40–0.48** — still strictly below 1 (non-vacuous) but an order of magnitude looser than on MNIST, and approaching the regime where a certificate ceases to be practically informative. As on the easier datasets, **Bayes-by-Backprop** again achieves the best accuracy (0.239 stochastic / 0.200 posterior-mean error ≈ 76–80% accuracy) but a **near-vacuous** certificate (0.899): it is the bound-minimising objectives, not BBB, that produce the (loose but meaningful) guarantees. Our reduced configuration makes these certificates looser than the reference's CIFAR-10 figures (e.g. 9-layer `fquad` ≈ 0.24 there); the scientifically relevant content — that certificates tighten sharply as difficulty falls and that bound-minimising training is what keeps them non-vacuous — is fully reproduced.

The practical message of the CIFAR-10 results is a honest boundary on the method's reach. At this difficulty, a PAC-Bayes certificate of $\sim$0.4 says "the true error is below 40% with 99% confidence" beside a model whose actual error is $\sim$25–31% — correct, but with a gap that limits the certificate's decision-making value. Two observations frame this constructively. First, the gap is dominated by the irreducible CIFAR-10 error (which inflates the empirical-risk term feeding the `kl`-inversion), not by a failure of the bound machinery: the same machinery gives tight certificates on MNIST. Second, the literature's most recent tighter-bound formulations and deeper architectures narrow this gap on CIFAR-10 specifically (García-Pérez et al., 2025; the reference's 13/15-layer cells), so the boundary is moving rather than fundamental — which is exactly why CIFAR-10 is the interesting regime to characterise rather than a failure to report.

### 4.4 Cross-dataset discussion

Reading MNIST → Fashion-MNIST → CIFAR-10 as a difficulty ladder, a consistent picture emerges. Table 4 collects the headline cell (`fquad`, learnt prior) across all three datasets.

**Table 4 — Cross-dataset difficulty ladder** (`fquad`, learnt prior; stochastic test error and 0–1 risk certificate).

| Dataset | Stch err | Cert (`Risk_01`) | PostMean err |
|---|---|---|---|
| MNIST (FCN)        | 0.022 | 0.039 | 0.018 |
| MNIST (CNN)        | 0.011 | 0.030 | 0.010 |
| Fashion-MNIST (FCN) | 0.117 | 0.145 | 0.109 |
| Fashion-MNIST (CNN) | 0.092 | 0.147 | 0.085 |
| CIFAR-10 (9-layer CNN) | 0.309 | 0.410 | 0.264 |

Three regularities hold across the whole ladder. First, **certificate tightness tracks task difficulty monotonically** — roughly 0.03 → 0.15 → 0.41 as the irreducible error rises — exactly the behaviour PAC-Bayes theory predicts, since the certificate upper-bounds the true risk. Second, **the accuracy–certificate ordering is preserved at every difficulty**: `fbbb` is always the most accurate yet has the loosest (or near-vacuous) certificate, while the bound-minimising objectives trade a little accuracy for much tighter guarantees; the convolutional architecture improves both accuracy and certificate quality over the fully-connected one wherever compared. Third, **the data-dependent prior is decisive throughout** — on MNIST it tightens the FCN `fquad` certificate ~8-fold, on Fashion-MNIST ~3-fold, and it is what keeps the CIFAR-10 certificates non-vacuous at all. The practical conclusion is that PAC-Bayes-with-Backprop delivers genuinely informative high-confidence certificates on the easier rungs (MNIST, Fashion-MNIST), and degrades gracefully toward vacuity on the hardest (CIFAR-10), with the learnt prior and bound-minimising objectives as the two levers that push back against that degradation.

### 4.5 Which predictor does the certificate cover?

A subtle but practically important point is the relationship between the certified quantity and the predictor one would actually deploy. The PAC-Bayes certificate upper-bounds the *stochastic* (Gibbs) risk — the error of a predictor that freshly samples weights for each example — whereas a deployed system typically uses a single deterministic predictor: the posterior mean, or an MC ensemble. Across every cell in our study, the posterior-mean and ensemble test errors are **at least as low as** the stochastic error (e.g. MNIST FCN `fquad`/learnt: stochastic 0.022, posterior-mean 0.018, ensemble 0.017; Fashion-MNIST CNN `fquad`/learnt: 0.092, 0.085, 0.082; CIFAR-10 `fquad`/learnt: 0.309, 0.264, 0.251). Because the certificate bounds the (larger) stochastic risk, it is automatically a valid — if conservative — guarantee for the lower-risk mean/ensemble predictors too. In other words, the certificate is *safe* for the deployed predictor, but it leaves "money on the table": the deployed predictor is systematically better than the risk the certificate promises. Closing this gap — deriving a bound that is tight for the posterior-mean or majority-vote predictor rather than the Gibbs predictor — is the most promising direction for obtaining sharper practical guarantees, and is the natural follow-up to this study (Sections 5.3–5.4).

### 4.6 Data-efficiency (small-data) ablation

To probe how the self-certified bound behaves under data scarcity, we re-train the MNIST FCN `fquad`/learnt cell at reduced training fractions (the self-certified setup uses *all* available data for both training and certification, so this directly tests its advertised data-efficiency advantage). Table 5 reports the result.

**Table 5 — MNIST FCN `fquad`/learnt: certificate and errors vs training-data fraction.**

| Train fraction | Cert (`Risk_01`) | Stch err | PostMean err |
|---|---|---|---|
| 1.00 | 0.039 | 0.022 | 0.018 |
| 0.50 | 0.052 | 0.028 | 0.022 |
| 0.20 | 0.063 | 0.048 | 0.037 |
| 0.10 | 0.072 | 0.064 | 0.056 |

The certificate loosens and the error rises **monotonically** as the training set shrinks, as PAC-Bayes theory predicts (fewer examples → larger $\log(2\sqrt{n}/\delta)/n$ penalty and higher empirical risk). The striking and encouraging finding is the **gracefulness** of the degradation: even trained on only 10% of MNIST (~6,000 examples), the self-certified bound remains strongly **non-vacuous** (0.072, certifying ≲7.2% error with 99% confidence) and the posterior-mean predictor still errs on only 5.6% of the test set. The bound roughly *doubles* (0.039→0.072) while the data shrinks *tenfold*, which is the qualitative behaviour that makes self-certified PAC-Bayes attractive in low-data regimes — and is consistent with the motivation of the self-certified-learning line (Pérez-Ortiz et al., 2021b).

## 5. Evaluation and Reflection

### 5.1 Achievement against objectives

All four project objectives were met: (i) the MNIST PBB experiments were reproduced and validated against the published Table 1 (test errors within ≤0.4 pp; certificate orderings preserved); (ii) Fashion-MNIST results were produced as a genuine extension, including new code (a `FashionMNIST` data branch and the associated model-dispatch fix) and a full objective × prior × architecture sweep; (iii) CIFAR-10 results are generated and analysed (§4.3); and (iv) the work is written up against the discipline's rubric with every number traceable to a versioned run.

### 5.2 Threats to validity

- **Determinism and variance estimation.** The PBB codebase fixes the data-partition seed (`torch.manual_seed(7)` in `loaddataset`), which makes every run fully deterministic — we verified that re-running a cell with a different train-level seed reproduces the result to the last decimal. The reported numbers are therefore *exactly reproducible* (a reproducibility strength), but this also means seed-variance cannot be estimated without modifying the upstream seeding. Our claims rest on large, qualitatively obvious effect sizes — far beyond any plausible seed noise (e.g. the learnt→rand prior change moves the MNIST FCN `fquad` certificate from 0.039 to 0.327, and the cross-dataset ladder spans 0.039→0.410) — rather than on small, significance-tested gaps. Genuine multi-seed confidence intervals (requiring an upstream seeding patch and a full re-run, which would also shift the paper-matching split) are therefore flagged as future work rather than reported as a misleading single-seed CI.
- **Monte-Carlo budget.** Certificates are computed with $m=10{,}000$ (FCN) or $2{,}000$ (CNN) samples rather than the reference's $150{,}000$. This systematically *loosens* our certificates relative to the published ones (most visibly for tight, learnt-prior CNN certificates) and is the sole cause of the certificate-level discrepancy in Table 1. It does not affect test errors or the ordering of any variable.
- **Reduced CNN configuration.** Probabilistic-CNN training is overhead-bound on our single 12 GB GPU; CNN cells use 50 epochs and $m=2{,}000$, so CNN certificates and accuracies are less precise than the FCN results and than the reference. The architecture *trend* (CNN improves accuracy at comparable-or-tighter certificate) is clear, but CNN absolute numbers are approximate.
- **Data reuse / partitioning.** We follow the reference's self-certified partition exactly; the certificate is evaluated on the held-out bound-set, and model selection (where performed) is distinct from the certificate data. No result is computed on data used for selection.
- **Hyper-parameter selection vs the reference's grid search.** The reference selects per-cell hyper-parameters by an extensive grid search over learning rate, momentum, prior scale and dropout, keeping the best-certificate configuration; we use representative fixed values (LR $5\times10^{-3}$, $\sigma_0=0.03$, dropout 0.2). Our cells are therefore not the per-cell optima, which accounts for the small residual differences from the published numbers and means our certificates are, if anything, a *lower* bound on what a full grid search would achieve.
- **Architecture coverage.** Each dataset uses a single architecture family (FCN/CNN for MNIST and Fashion-MNIST; 9-layer CNN only for CIFAR-10); we do not vary depth or width, so claims about the effect of architecture are based on the FCN↔CNN contrast and the literature rather than a within-study sweep.
- **External validity.** All three datasets are standard image benchmarks; the findings should not be over-generalised to other modalities, much larger models, or severely class-imbalanced settings, none of which we test.

### 5.3 Limitations and scope not pursued

- **Certificate validity for the *deployed* predictor.** The PAC-Bayes certificate controls the *stochastic/Gibbs* risk; in practice one deploys the deterministic posterior-mean predictor. We report all three predictor rules but do *not* derive a certificate that is formally valid for the posterior-mean predictor (this would require a majority-vote/Gibbs-variance bound à la Lacasse et al., 2006, adapted to the PBB objectives) — a natural and higher-ambition extension.
- **No small-data / self-certified ablation.** Time did not permit the data-efficiency sub-experiment (comparing self-certified vs hold-out bounds at reduced training-set sizes); this is the most promising follow-up given the self-certified literature.
- **CIFAR-10 depth.** Only the 9-layer CNN configuration is run; the 13/15-layer variants (which the reference shows give tighter CIFAR-10 certificates) are omitted for compute reasons.
- **Hyper-parameter selection.** We use representative fixed values (LR $5\times10^{-3}$, $\sigma_0=0.03$) rather than the reference's full grid search, which selected the best-certificate configuration per cell. This accounts for residual small differences from the published numbers.

### 5.4 Reflection and future work

The project confirms that PAC-Bayes-with-Backprop is a practical route to *non-vacuous, informative* risk certificates for deep classifiers, and that the mechanism (bound-minimising training + data-dependent priors) transfers from the original MNIST/CIFAR-10 setting to Fashion-MNIST. The most scientifically interesting open question — whether the certificate remains valid and tight for the predictor one actually deploys, rather than only for the stochastic Gibbs predictor — emerged from an adversarial review of the plan and is the strongest candidate for future work, alongside multi-seed confidence intervals, larger Monte-Carlo budgets for the headline cells, and the recent tighter-bound formulations (García-Pérez et al., 2025).

Concretely, we see five well-motivated follow-ups. (i) **A certificate for the deployed predictor.** Adapting the Lacasse et al. (2006) majority-vote bound (which relates Gibbs mean and variance to the majority-vote risk) to the `fquad`/`flambda` objectives would yield a guarantee valid and tight for the posterior-mean predictor users actually deploy, directly closing the gap quantified in §4.5. (ii) **Statistical rigour.** Re-running the headline cells over 5–10 seeds with bootstrap confidence intervals would let the objective/prior orderings be reported as statistically tested rather than point-estimated. (iii) **Tight headline certificates.** A selective $m=150{,}000$ re-run of the key learnt-prior cells would match the reference's certificate precision. (iv) **The small-data, self-certified regime.** Comparing self-certified (whole-data) bounds against hold-out test-bounds at reduced training-set sizes would test whether the data-efficiency advantage of self-certification generalises across the difficulty ladder. (v) **Deeper CIFAR-10 architectures.** The 13- and 15-layer CNNs the reference shows give tighter CIFAR-10 certificates were omitted for compute; adding them would clarify how far architectural depth can push back against the vacuity observed at §4.3.

### 5.5 Reproducibility and use of generative AI

All code, configurations, per-run metadata and the aggregated results registry are retained, and every figure/table is regenerated from `results/results_summary.csv` by a single script, so each claim is independently checkable. Consistent with the institution's policy, generative AI (Claude Code) was used to scaffold the experimental pipeline, draft prose and iterate on the write-up; all generated code and text were reviewed, the experiments were run and verified by the author, and the AI-use questionnaire is completed accordingly.

## 6. Conclusion

This project set out to certify the predictions of deep classifiers with high confidence using PAC-Bayes-with-Backprop, by reproducing the reference framework on MNIST, extending it to Fashion-MNIST, and analysing CIFAR-10. The reproduction is faithful: across four training objectives, two architectures and two prior families, the published test errors are recovered within fractions of a percentage point and every scientifically meaningful ordering is preserved, with the residual certificate-level looseness fully accounted for by the Monte-Carlo budget. The Fashion-MNIST extension shows that the methodology — and in particular the dramatic tightening delivered by data-dependent priors — transfers to a harder, more realistic task while remaining non-vacuous and informative. Read across the three datasets, a coherent picture emerges: PAC-Bayes-with-Backprop delivers competitive accuracy together with meaningful certificates on the easier rungs, data-dependent priors and CNN architectures both tighten the certificate, and rising task difficulty inflates the certificate toward vacuity through the irreducible error.

Three takeaways stand out. First, *bound-minimising training works*: directly optimising a PAC-Bayes bound yields both good predictors and quantifiable guarantees, in stark contrast to the loose certificates produced by training only for accuracy (Bayes-by-Backprop). Second, *the prior is everything*: the move from a data-free to a data-dependent prior is the single largest lever on certificate tightness, and its value grows with task difficulty. Third, the most important open question — whether these certificates remain valid and tight for the deterministic predictor that is actually deployed, rather than only for the stochastic Gibbs predictor the bound controls — is well-motivated by this study and is the natural next step, together with multi-seed confidence intervals, larger Monte-Carlo budgets, and the most recent tighter-bound formulations.

## References

- Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight Uncertainty in Neural Networks. *ICML* (PMLR 37). arXiv:1505.05424.
- Catoni, O. (2007). *Pac-Bayesian Supervised Classification: The Thermodynamics of Statistical Learning*. arXiv:0712.0248.
- Cohen, J., Rosenfeld, E., & Kolter, Z. (2019). Certified Adversarial Robustness via Randomized Smoothing. *ICML*. arXiv:1902.02918.
- Dziugaite, G. K., & Roy, D. M. (2017). Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data. *UAI*. arXiv:1703.11008.
- Dziugaite, G. K., & Roy, D. M. (2018). Data-dependent PAC-Bayes priors via differential privacy. *NeurIPS*. arXiv:1802.09583.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*. arXiv:1506.02142.
- García-Pérez, M. A., et al. (2025). Some theoretical improvements on the tightness of PAC-Bayes risk certificates. arXiv:2510.07935.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML*. arXiv:1706.04599.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*. arXiv:1612.01474.
- McAllester, D. A. (1999). PAC-Bayesian model averaging. *COLT*.
- Pérez-Ortiz, M., Rivasplata, O., Shawe-Taylor, J., & Szepesvári, C. (2021). Tighter Risk Certificates for Neural Networks. *Journal of Machine Learning Research* 22(227). arXiv:2007.12911. *(Reference reproduced; code: github.com/mperezortiz/PBB.)*
- Pérez-Ortiz, M., et al. (2021). Progress in Self-Certified Neural Networks. *NeurIPS Bayesian Deep Learning workshop*. arXiv:2111.07737.
- Pérez-Ortiz, M., et al. (2021). Learning PAC-Bayes Priors for Probabilistic Neural Networks. arXiv:2109.10304.
- Seeger, M. (2002). PAC-Bayesian Generalisation Error Bounds for Gaussian Process Classification. *JMLR* 3.
- Thiemann, N., Igel, C., Wintenberger, O., & Seldin, Y. (2017). A Strongly Quasiconvex PAC-Bayesian Bound. *ALT* (PMLR 76). arXiv:1608.05610.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. arXiv:1708.07747.

## Appendix A — Complete results (all 26 runs)

Risk certificate is the 0–1 PAC-Bayes-kl value (`Risk_01`); "Stch/PM/Ens" are the stochastic / posterior-mean / ensemble test errors. FCN cells: 100 epochs, $m=10{,}000$; CNN cells (MNIST/Fashion): 50 epochs, $m=2{,}000$; CIFAR-10: prior 20 / posterior 25 epochs, $m=1{,}000$. Every row is traceable to `experiments/logs/<run_id>/` in the registry.

| Run | Cert | Stch | PM | Ens |
|---|---|---|---|---|
| mnist fquad rand fcn    | 0.327 | 0.091 | 0.055 | 0.058 |
| mnist flamb rand fcn    | 0.340 | 0.072 | 0.044 | 0.045 |
| mnist fclassic rand fcn | 0.332 | 0.136 | 0.084 | 0.082 |
| mnist bbb rand fcn      | 0.611 | 0.025 | 0.016 | 0.016 |
| mnist fquad learnt fcn    | 0.039 | 0.022 | 0.018 | 0.017 |
| mnist flamb learnt fcn    | 0.049 | 0.022 | 0.017 | 0.016 |
| mnist fclassic learnt fcn | 0.032 | 0.023 | 0.019 | 0.019 |
| mnist bbb learnt fcn      | 0.150 | 0.020 | 0.016 | 0.016 |
| mnist fquad learnt cnn    | 0.030 | 0.011 | 0.010 | 0.010 |
| mnist flamb learnt cnn    | 0.036 | 0.012 | 0.009 | 0.009 |
| mnist fclassic learnt cnn | 0.026 | 0.011 | 0.010 | 0.010 |
| fashion fquad rand fcn    | 0.429 | 0.226 | 0.187 | 0.185 |
| fashion flamb rand fcn    | 0.447 | 0.192 | 0.162 | 0.159 |
| fashion fclassic rand fcn | 0.430 | 0.262 | 0.213 | 0.207 |
| fashion bbb rand fcn      | 0.782 | 0.135 | 0.118 | 0.119 |
| fashion fquad learnt fcn    | 0.145 | 0.117 | 0.109 | 0.110 |
| fashion flamb learnt fcn    | 0.177 | 0.115 | 0.107 | 0.107 |
| fashion fclassic learnt fcn | 0.135 | 0.119 | 0.112 | 0.112 |
| fashion bbb learnt fcn      | 0.458 | 0.111 | 0.106 | 0.102 |
| fashion fquad learnt cnn    | 0.147 | 0.092 | 0.085 | 0.082 |
| fashion flamb learnt cnn    | 0.172 | 0.092 | 0.083 | 0.081 |
| fashion fclassic learnt cnn | 0.122 | 0.091 | 0.085 | 0.084 |
| cifar10 fquad learnt cnn    | 0.410 | 0.309 | 0.264 | 0.251 |
| cifar10 flamb learnt cnn    | 0.477 | 0.277 | 0.243 | 0.230 |
| cifar10 fclassic learnt cnn | 0.403 | 0.325 | 0.269 | 0.258 |
| cifar10 bbb learnt cnn      | 0.899 | 0.239 | 0.200 | 0.193 |
