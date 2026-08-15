# Phase 05 Notebook Kernel Mismatch — 2026-08-12

## Error

The first offline notebook execution failed in Phase 1 while importing
PyTorch. `python -m jupyter nbconvert` resolved the `jupyter-nbconvert`
entrypoint from Anaconda Python 3.13 instead of the repository `.venv`.

## Phase

Execution verification from Phase 0 through Phase 6; failure occurred before
the Phase 5 enhancement cell ran.

## Context

- Required environment: repository `.venv`, Python 3.11
- Incorrect runtime: `/opt/anaconda3/bin/python`, Python 3.13
- Offline cache policy remained enabled

## Root Cause

The `jupyter` launcher imported an external Anaconda nbconvert entrypoint even
though the command began with the `.venv` Python executable. The default
`python3` kernelspec also pointed to Anaconda.

## Evidence

The traceback loaded `torch` from
`/opt/anaconda3/lib/python3.13/site-packages/torch` and failed with an unresolved
`libtorch_cpu.dylib` symbol. Inspection confirmed that the existing
`venv-practice-2` kernelspec points to the repository `.venv` Python 3.11.

## Resolution

Executed nbconvert directly as the `.venv` module and selected the existing
`.venv` kernelspec:

```bash
../../.venv/bin/python -m nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=venv-practice-2 \
  notebook_practice_3/practice_3.ipynb
```

No dependency, dataset, model or methodology was changed.

## Final Status

RESOLVED. All seven code cells executed with counts 1–7, no error outputs;
Phase 5 and Phase 6 both reported PASS.

