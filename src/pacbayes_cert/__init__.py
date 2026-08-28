"""pacbayes_cert: a clean, auditable PAC-Bayes risk-certificate pipeline.

Re-implements the PBB experiments of Perez-Ortiz et al. (2021) with corrected
sample accounting (n_bound), honest joint-confidence (delta_mc + delta_pac),
batching-invariant Monte-Carlo aggregation, genuine multi-seed control, and
explicit, validated data splits with full provenance.
"""
from .config import ExpConfig
from .schema import RunResult, SCHEMA_VERSION

__all__ = ["ExpConfig", "RunResult", "SCHEMA_VERSION"]
__version__ = "2.0.0"
