# `scripts/legacy/` — superseded, not part of the reported pipeline

Nothing in this directory produces any number, table or figure in the
dissertation. It is the pre-correction tooling from the exploratory phase, kept
so the audit trail is complete, and it is retained rather than deleted precisely
because the dissertation's argument is about what the earlier pipeline got wrong.

Two things here are actively misleading if read as current:

- **`make_figures_hardcoded.py`** does what its name says. It was the reason the
  figure step was rewritten: the current `scripts/make_figures.py` reads
  `results/processed/` and hard-codes nothing, which is what makes the claim in
  Appendix~B ("no reported number is entered by hand") true. Do not run this one.
- **Absolute remote paths** (`/root/autodl-tmp/...`) point at a rented GPU box
  used early on and no longer in use. Every reported run is local; see
  `results/raw/*/environment.json`.

The pipeline that produced the reported results is, in order:
`run.py`, `scripts/run_matrix.sh`, `scripts/prior_fit_on_s0.py`,
`scripts/aggregate.py`, `scripts/gen_tables.py`, `scripts/make_figures.py`.
