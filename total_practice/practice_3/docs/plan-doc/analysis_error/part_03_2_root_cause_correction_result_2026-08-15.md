# PART 3.2.E — Root-cause Correction Diagnosis Result

Date: 2026-08-15  
Status: root-cause category and concrete fault identified; final fix not implemented.

## Known-easy fixture — PASS

The temporary fixture contained 64 unique synthetic texts, balanced 32 negative and 32 positive, with explicit repeated sentiment signals. Fixture SHA-256:

`6a7404dfd06685fb4edae0f2f4cc8ead56bbbff700e43dc1f1577ffb8af8b514`

Both branches used CPU, seed 42, the same fresh-model fingerprint, a frozen/eval DistilBERT backbone, dynamic padding, batch 64, AdamW, constant LR `1e-3`, and 100 steps. No Validation, Holdout, or Test data was used.

## Discovery: tokenizer resolution is the concrete fault

The first A/B unexpectedly failed in both the direct loop and Trainer. A feature-level audit then showed:

- all 64 distinct texts produced only **one unique input-ID row**;
- negative and positive inputs both became `[CLS] [UNK] ... [SEP]`;
- class-centroid L2 distance was approximately `2.89e-6`;
- centroid cosine similarity was approximately `1.0000002`.

The local cache contains two different identifiers:

| Identifier | Tokenizer class | Vocab size | Result |
|---|---|---:|---|
| `distilbert/distilbert-base-uncased` | `BertTokenizer` | **5** | every normal word becomes `[UNK]` |
| `distilbert-base-uncased` | `BertTokenizer` | **30,522** | normal WordPiece IDs |

The namespaced cache snapshot contains model/config files but lacks `tokenizer.json`, `tokenizer_config.json`, and `vocab.txt`. `AutoTokenizer.from_pretrained(..., local_files_only=True)` does not fail closed; in this environment it constructs a five-special-token tokenizer.

The v2 runner maps the frozen tokenizer name `distilbert-base-uncased` through the same model alias to `distilbert/distilbert-base-uncased`. Consequently the official run received no lexical information. The existing pretraining checks validated fields, lengths and labels but did not assert vocabulary size, known-token coverage, or `[UNK]` rate. This allowed the broken tokenizer to pass readiness.

This fully explains:

- Train/eval loss near `ln(2) = 0.6931`;
- class-0 collapse;
- nonzero gradients and changed weights without useful learning;
- failure on CPU and MPS;
- failure under larger LR and head-only controls.

## Direct PyTorch loop with broken tokenizer

- Initial loss: `0.693620`
- Final loss: `0.693209`
- Initial accuracy: `0.50`
- Final accuracy: `0.50`
- Final prediction counts: class 0 = 64, class 1 = 0
- Overfit >=90%: **NO**

## HF Trainer with broken tokenizer

- Initial loss: `0.693620`
- Final loss: `0.693179`
- Initial accuracy: `0.50`
- Final accuracy: `0.50`
- Final prediction counts: class 0 = 0, class 1 = 64
- Overfit >=90%: **NO**

The failures were materially equivalent, rejecting Trainer as the unique cause.

## Controlled confirmation with complete tokenizer cache

The A/B was repeated without changing model weights or training configuration; only the temporary tokenizer source used the complete 30,522-token cache. Initial model fingerprints remained identical:

`3fc8fe93a3d0149b5f030e94fcf7d773764376f0b6bf2c54c60e9a6c25a8b103`

### Direct PyTorch loop

- Initial loss: `0.691022`
- Final loss: `0.000112`
- Initial accuracy: `0.50`
- Final accuracy: **1.00**
- Final prediction counts: class 0 = 32, class 1 = 32
- Initial classifier/pre-classifier gradient norms: `0.35116 / 0.66120`
- Final classifier/pre-classifier gradient norms: `0.001286 / 0.000188`
- Classifier delta L2: `0.500740`
- Overfit >=90%: **YES**

### Hugging Face Trainer

- Initial loss: `0.691022`
- Final loss: `0.0000144`
- Initial accuracy: `0.50`
- Final accuracy: **1.00**
- Final prediction counts: class 0 = 32, class 1 = 32
- Initial classifier/pre-classifier gradient norms: `0.35116 / 0.66120`
- Final classifier/pre-classifier gradient norms: `0.000181 / 0.0000416`
- Classifier delta L2: `0.569142`
- Overfit >=90%: **YES**

Both branches learn immediately once lexical token IDs are restored. Their final loss differs but both decisively satisfy the intended functional criterion; this is not a material behavioral failure.

## Root-cause category

**Model-data pipeline — tokenizer alias/cache resolution and missing tokenizer integrity guards.**

Not supported as the main cause:

- Hugging Face Trainer integration;
- MPS backend;
- LR strength;
- optimizer coverage;
- frozen parameters;
- label pipeline;
- general incompatibility of the installed environment stack.

## Environment audit

| Component | Installed |
|---|---|
| Python | 3.11.14 |
| torch | 2.13.0 |
| transformers | 5.14.1 |
| datasets | 5.0.1 |
| accelerate | 1.14.0 |
| tokenizers | 0.22.2 |

Local package metadata declares Transformers compatibility with tokenizers `>=0.22.0, <=0.23.0`, torch `>=2.4`, accelerate `>=1.1.0`, and datasets `>=2.15.0`; the installed versions satisfy these bounds. More importantly, both direct PyTorch and Trainer reached 100% using the corrected temporary tokenizer source in this unchanged environment.

**Stable environment revision recommended: NO.** A separate MPS SDPA/dropout limitation was observed in a prior frozen-backbone diagnostic, but it did not cause the official full fine-tuning failure and does not justify changing the whole environment for this correction.

## Recommended correction for a future `practice_3_v2.1` plan — not implemented

1. Stop using the model-cache alias blindly for tokenizer loading.
2. Resolve tokenizer to an artifact that actually contains the canonical 30,522-token vocabulary, or download/verify the complete namespaced tokenizer snapshot.
3. Add fail-closed preflight assertions before training:
   - expected tokenizer class/family;
   - vocab size 30,522;
   - canonical tokens such as `this`, `movie`, `good`, `bad` do not map to `[UNK]`;
   - bounded `[UNK]` ratio over deterministic Train samples;
   - more than one unique encoded sequence for distinct sample texts.
4. Re-run a known-easy direct/Trainer smoke diagnostic after correction.
5. Create a new protocol version and preserve `p3v2_lr_2e-5` as invalid-pipeline evidence; do not mutate or reset v2.0.

## Final classification

- Known-easy fixture: PASS
- Direct PyTorch overfit: YES
- HF Trainer overfit: YES
- Direct versus Trainer materially different: NO
- Root cause identified: YES
- Root cause category: Model-data pipeline
- Concrete cause: five-token all-`[UNK]` tokenizer loaded from incomplete namespaced cache
- Final fix implemented: false

## Integrity record

- Official protocol modified: false
- Real registry modified: false
- `p3v2_lr_2e-5` preserved: true
- `p3v2_lr_3e-5` executed: false
- `p3v2_lr_5e-5` executed: false
- Holdout accessed: false
- Official Test accessed: false
