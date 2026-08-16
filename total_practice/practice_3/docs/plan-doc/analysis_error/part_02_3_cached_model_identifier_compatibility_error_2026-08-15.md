# Part 2.3 — Cached Model Identifier Compatibility Error

## Error

Offline `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")` failed because the local cache stores the same checkpoint under its canonical Hub identifier `distilbert/distilbert-base-uncased`.

## Context

The frozen v2 run config records the legacy short identifier. Practice 3 Phase 7 and its verified artifact use the canonical organization-qualified identifier.

## Evidence

- Integration stopped before model/Trainer/optimizer construction.
- Error: no `pytorch_model.bin` or `model.safetensors` resolved for the short offline alias.
- Phase 7 source: `MODEL_CHECKPOINT = "distilbert/distilbert-base-uncased"`.
- No readiness manifest was created.

## Root cause

Hub alias resolution is unavailable in offline mode and the cache key is organization-qualified. This is identifier resolution, not a weight/model difference.

## Safe resolution

Keep the frozen run-config field unchanged and resolve only this exact approved alias to the canonical cached Hub ID at load time. Record both requested and resolved identifiers in readiness evidence. Reject every other mismatch. Verify DistilBERT model type, 2 labels and tokenizer family.

## Methodology impact

None: same pretrained DistilBERT checkpoint, tokenizer contract and weights; no new model or fallback.

## Status

FIX APPROVED BY EXISTING PROTOCOL EQUIVALENCE — implementation required, then rerun zero-training integration.

