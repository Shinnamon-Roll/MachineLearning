# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Binary image classifier (Salmon vs Trout) with two independent PyTorch models and a Next.js dashboard that runs both models side by side.

- `model-1/` — `ImprovedDenseNet121`, single-phase training, CrossEntropy loss.
- `model-2/` — `CustomMobileNetV2`, 2-phase training (frozen head, then full fine-tune at `lr * 0.1`), Focal Loss, CLAHE preprocessing.
- `dashboard/` — Next.js 16 (App Router, React 19, Tailwind 4, Recharts).
- `Docs/` — course grading rubric (Thai), not code.

The two model directories are self-contained and duplicate code (`data_loader.py`, `evaluate.py`, `inference.py`) on purpose. Each script imports siblings by bare name (`from model import ...`), so run Python scripts from inside their own model directory.

## Commands

```bash
# Python (per model)
pip install -r model-1/requirements.txt   # model-2 also needs opencv-python-headless
cd model-1 && python train.py             # writes salmon_trout_binary_model.pth + dashboard JSON
cd model-2 && python train.py             # writes mobilenet_v2_best.pth + dashboard JSON
cd model-1 && python test_setup.py        # only test: model instantiation + freeze_layers smoke check
python model-2/inference.py <image> [--model_path ...]   # prints one JSON line

# Dashboard
cd dashboard && npm install
npm run dev     # http://localhost:3000
npm run build
npm run lint
```

## Architecture: how the pieces connect

**Inference path.** `dashboard/src/app/api/predict/route.ts` saves the upload to `dashboard/public/uploads/`, then shells out to `python3 ../model-1/inference.py` and `python3 ../model-2/inference.py` (paths resolved from `process.cwd()`, so `npm run dev` must run inside `dashboard/`). Each `inference.py` must print exactly one JSON object to stdout — `{class, confidence, probabilities: {Salmon, Trout}}` or `{error}`. Any extra `print` in the inference path breaks `JSON.parse`. The route expects weights at `model-1/salmon_trout_binary_model.pth` and `model-2/mobilenet_v2_best.pth`; if missing, it returns `{error: "Model weights not found"}` for that model instead of failing.

**Metrics path.** Training/evaluation scripts write JSON into `dashboard/public/data/`, which `page.tsx` fetches on load:
- `metrics.json`, `training_history.json` (model 1)
- `metrics_model2.json`, `training_history_model2.json` (model 2)

Changing the shape of these JSON files requires matching changes in `dashboard/src/app/page.tsx`.

**Preprocessing must match between training and inference.** Model 2 applies `CLAHETransform` in train, val/test, and `inference.py` (the class is duplicated in `data_loader.py` and `inference.py`). Model 1 uses no CLAHE. Both normalize with ImageNet mean/std at 224x224. Class index order is fixed: `0 = Salmon`, `1 = Trout`.

**Dataset.** Not in git (`Image/`, `data/`, `dataset/` are gitignored, as are `*.pth`). Splits come from pre-split folders, not random splitting:
```
<DATA_DIR>/Salmon!/{Salmon Train, Salmon valid, Salmon Test}
<DATA_DIR>/Trout!/{Trout train, Trout valid, Trout test}
```
Folder names are case-sensitive and inconsistent between classes; missing folders only print a warning.

## Gotchas

- `DATA_DIR` and some output paths are hardcoded absolute paths under `/Users/shinnamon/Documents/Project/MachineLearning/` in `train.py` and `evaluate.py` of both models. The repo now lives at `/Users/shinnamon/project/MachineLearning/`, so update these before running.
- `train.py` writes dashboard JSON via relative `../dashboard/public/data/...` (requires cwd = model dir); `evaluate.py` uses absolute paths and loads weights via `model-N/...` (requires cwd = repo root). The two scripts assume different working directories.
- `model-1/test_setup.py` builds the model with `num_classes=6` (legacy); the real model uses 2.
- Device selection everywhere: MPS, then CUDA, then CPU.
