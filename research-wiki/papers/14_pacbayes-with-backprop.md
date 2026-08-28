# PAC-Bayes with Backprop

- **Authors:** Omar Rivasplata, Vikram M. Tankasali, Csaba Szepesvári
- **Venue/Year:** arXiv:1908.07380 (cs.LG/stat.ML) (2019)
- **ID:** arXiv:1908.07380 — ✓ verified
- **Contribution:** First statement of the 'PAC-Bayes with Backprop' (PBB) family: train a probabilistic net (Gaussian posterior over weights) by directly minimising a PAC-Bayes bound objective via backprop, using the binary 0-1 indicator loss made differentiable. Reports ~1.4% test error on MNIST with a ~2.3% non-vacuous risk certificate, demonstrating self-bounding learning without data splitting. ID verified via arXiv abstract page.
- **Relevance to our RQ:** Methodological foundation of the entire project. Provides the concrete MNIST numbers (test ~1.4%, cert ~2.3%) that anchor our reproduction baseline, and the two-objective PBB recipe (classical vs novel bound) that Pérez-Ortiz 2021 extends to three (fclassic/flambda/fquad). Establishes the objective family and self-certified-learning framing as the authors' own prior art -- our extension must be framed as a systematic study, not a method invention. Directly motivates H2 (self-certified vs hold-out).
