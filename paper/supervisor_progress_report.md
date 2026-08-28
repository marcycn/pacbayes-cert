**Subject:** MSc dissertation progress — full draft complete, two points I would value your view on

Dear Dr Rivasplata,

A short update on the dissertation. The full draft is now complete at 45 pages
and 8,948 words, and every experiment in it has been run and re-verified. I would
be grateful for your comments, and there are two specific points I would like your
view on before I submit.

**Where the work stands.** I re-implemented the PAC-Bayes-with-Backprop
computation from scratch, reproduced four MNIST cells of Pérez-Ortiz et al. (2021)
under a reduced-compute protocol, and extended the same pipeline to
Fashion-MNIST and CIFAR-10. The reproduction matches the published stochastic
test errors to within 0.0033 on all four target cells; our certificates are looser
by +0.0214 to +0.0423, and re-drawing at the reference's full Monte-Carlo budget
recovers a little over half of that gap on the checkpoint I examined. The
remainder I leave unattributed rather than explained away. The learnt-prior MNIST
certificates are non-vacuous (f_classic 0.0455, f_quad 0.0580) and loosen to
0.1746 on Fashion-MNIST and 0.4277 on CIFAR-10.

**What I take to be the main contribution.** Auditing the reference
implementation turned up five issues in the certificate pipeline, each of which I
corrected under a unit test. Three change the reported certificate — the
bound-set sample count, the batched Monte-Carlo aggregation, and the joint
confidence accounting, where reusing a single δ across both inversions
double-counts one failure probability. One affects reproducibility (seed control).
The fifth decides admissibility: a class-stratified partition sets the prior
subset's quotas from label counts taken over all of S, so the prior comes to
depend on the labels of the bound set even though the two subsets share no
example. I replaced it with a label-independent partition and verified all 115
saved partitions by recomputing them from their recorded seeds and comparing
index for index.

The suite now stands at 42 tests, all passing. The registry holds 115 runs, of
which 113 are reported; the two excluded are learnt-prior runs whose prior network
never left the chance-level plateau (0.866 and 0.899 error on S_0). Every reported
number is regenerated from that registry, and I re-check the hand-typed numbers in
the prose against recomputation — currently 25 of 25 agree.

**The two points I would value your view on.**

First, the prior scale. σ_0 was settled by comparing certificates computed on the
bound set, which (2) does not licence, so every certificate now carries a union
over a data-independent grid of candidate scales, Σ = {0.001, …, 0.999}, at
δ_PAC/K with K = 999. What I cannot demonstrate is that my search was confined to
Σ in advance: I wrote Σ down afterwards, and a procedure returning 0.0325 would
not have been covered. I therefore state the certificates as holding under the
assumption that only a three-decimal scale could have been produced, and carry the
gap as a threat to validity rather than a solved problem. I would welcome your
view on whether that is the right way to present it, or whether you would handle
the selection differently.

Second, scope. I certify only the Gibbs classifier. The posterior-mean and
finite-ensemble predictors attain lower test error, but I report those as
descriptive deployment metrics and say explicitly that the guarantee does not
transfer; a majority-vote guarantee would need the C-bound, which I have not
computed. I would like to know whether you think leaving that uncomputed is
acceptable at this scope, or whether it is worth attempting in the time remaining.

I am happy to send the PDF, or to meet at whatever time suits you.

With thanks for your guidance throughout,

[your name]
Student ID 14215174
