# On Calibration of Modern Neural Networks

- **Authors:** Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger
- **Venue/Year:** ICML 2017 (PMLR vol. 70, pp. 1321-1330; arXiv:1706.04599) (2017)
- **ID:** arXiv:1706.04599 — ✓ verified
- **Contribution:** Demonstrates modern deep nets are mis-calibrated (confidence != accuracy) and define Expected Calibration Error (ECE); proposes single-parameter post-hoc temperature scaling that recalibrates softmax confidences without retraining, outperforming Platt scaling/isotonic regression.
- **Relevance to our RQ:** Calibration baseline central to 'high-confidence predictions'. Our certificates are distribution-free PAC-Bayes bounds; we should report ECE alongside and argue a certificate is a stronger guarantee than a calibrated point estimate. Temperature scaling is a cheap baseline to beat on the accuracy-confidence tradeoff across MNIST/FMNIST/CIFAR-10 (H3).
