"""Phase 52 — Attention Extraction.

This package implements Phase 52 of the COURSE_WORK scientific pipeline.
Phase 52 is the **extraction + integrity** phase for temporal self-attention.

It does NOT interpret attention, train, fit scaler, select best seed, or
create an ensemble.

Sub-phases:
    52-A: preflight + source verification + architecture amendment
    52-B: extraction contract freeze + infrastructure
    52-C: official raw attention extraction
    52-D: integrity audits
    52-E: derived summaries
    52-F: Phase 53-57 handoffs
    52-G: tests + summary + report + README + signoff
    52-H: notebook visualization (DEFERRED)

All modules in this package MUST obey the canonical Phase 52 boundary:

    - EXTRACTION + INTEGRITY ONLY
    - No training / fine-tune / backward / scaler fitting
    - Strict-load only the three frozen final Transformer checkpoints
    - eval + torch.inference_mode forward passes only
    - raw attention stored as float32 with no averaging
    - Same dense case set for all three seeds
"""

__version__ = "ATTENTION_EXTRACTION-v1"
__phase__ = 52
