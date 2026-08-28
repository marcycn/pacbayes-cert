# Tighter Risk Certificates for Neural Networks -- Official Code Repository (PBB)

- **Authors:** María Pérez-Ortiz, Omar Rivasplata, John Shawe-Taylor, Csaba Szepesvári (repository owner: mperezortiz)
- **Venue/Year:** GitHub (companion to JMLR 2021 paper) (2021)
- **ID:** https://github.com/mperezortiz/PBB — ✓ verified
- **Contribution:** Official reference implementation of PAC-Bayes with Backprop. Contains the training code for the fclassic, flambda, fquad (and fbbb, ferm) objectives, the data-free (Rand.Init) and data-dependent (Learnt) prior machinery, the risk-certificate computation via PAC-Bayes-kl inversion (delta=0.025), and the experiment configurations for MNIST FCN/CNN and CIFAR-10 deep CNNs reported in Tables 1, 2, and 5 of the paper.
- **Relevance to our RQ:** The code we must run and extend for the MNIST reproduction (Task #9), the Fashion-MNIST + CIFAR-10 extension (Task #10), and all ablations (Task #11). Verified as the genuine author repo; ground truth for matching Table 1/2/5 numbers. Sibling repo mperezortiz/pacbayespriors covers the related AAAI'22 'Learning PAC-Bayes priors' work relevant to H4.
