# Phase 1 — Environment Setup Log

**Date:** 2026-07-24  
**Agent:** Main Agent (Cursor)

## Environment Check

```bash
$ python3 --version
Python 3.10.11

$ python3 -c "import torch; print(torch.__version__)"
2.13.0

$ python3 -c "import torchvision; print(torchvision.__version__)"
0.28.0

$ python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), 'MPS:', torch.backends.mps.is_available())"
CUDA: False  MPS: True
```

## Installed Packages

| Package    | Version |
|------------|---------|
| Python     | 3.10.11 |
| PyTorch    | 2.13.0  |
| TorchVision| 0.28.0  |
| Matplotlib | 3.10.8  |
| NumPy      | 2.2.6   |
| Pandas     | 2.3.3   |
| Seaborn    | 0.13.2  |

## Files Created

- `processing_own_phase/requirements.txt`
- `processing_own_phase/__init__.py`
- `processing_own_phase/config.py` — central configuration, paths, class names, experiment definitions
- `processing_own_phase/utils.py` — seed, device, time-formatting, parameter-counting helpers

## Issues Encountered

1. **Matplotlib cache directory not writable** — `.matplotlib` and `.fontconfig` returned "not writable".
   - Resolved by setting `MPLCONFIGDIR` and `XDG_CACHE_HOME` to a writable directory inside `outputs/`.
2. **scikit-learn import failed** — `scipy.sparse.linalg._propack` had a corrupted Mach-O segment.
   - Resolved by removing the dependency and implementing confusion matrix manually with NumPy.
3. **Torchvision requires old transform API** — `torchvision.transforms` no longer has `ToImage()` / `ToDtype(scale=True)`.
   - Resolved by switching to `transforms.ToTensor()` which already produces float32 in [0, 1].

## Outcome

Environment is ready. Next: data pipeline.
