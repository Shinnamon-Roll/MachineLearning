# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Controlled comparison of two backbones for binary image classification (Salmon vs Trout): DenseNet121 and MobileNetV2. Both are trained with the **identical** recipe from `ml/config.py`; the backbone is the only variable. Don't introduce per-model training differences (epochs, loss, augmentation, preprocessing, scheduler, latency protocol) — that breaks the comparison.

- `ml/` — all Python: config, models, data/splits, training, evaluation, inference.
- `dashboard/` — Next.js 16 (App Router, React 19, Tailwind 4, Recharts); shows metrics and runs both models on an uploaded image.
- `Image/Salmon`, `Image/Trout` — dataset (gitignored, not pre-split).
- `Docs/` — course grading rubric (Thai), not code.

## Commands

```bash
pip install -r ml/requirements.txt
python ml/test_setup.py                      # smoke checks: forward shape, backbone freezing, split integrity
python ml/train.py --model all --quick       # 1+1 epochs, one seed — use to verify changes
python ml/train.py --model all               # full run: 3 seeds x 2 models x 30 epochs (hours on MPS)
python ml/evaluate.py --model densenet       # re-evaluate saved weights on the test split
python ml/inference.py <image> --model mobilenet

cd dashboard && npm install
npm run dev | npm run build | npm run lint
```
Python scripts resolve paths from their own location, so they run from any cwd.

## Architecture

**Recipe** (`ml/config.py` `CONFIG`): phase 1 trains only the head with the backbone frozen (`set_backbone_trainable` in `ml/models.py`, via each model's `.backbone`); phase 2 unfreezes everything at `lr * finetune_lr_factor`. Focal Loss (alpha from train class ratio), ReduceLROnPlateau on val loss recreated per phase, best weights chosen by val acc across both phases. `--quick` only shrinks epochs/seeds; the config actually used is saved into the metrics JSON.

**Splits**: `ml/data.py` `make_splits()` writes `ml/splits.json` once (stratified 80/10/10, fixed seed, byte-identical images grouped into one split). It is committed and shared by every model and seed. Delete it only if the dataset changes. `eval_transform()` is used for val, test and inference, so CLAHE preprocessing always matches training.

**Model registry**: `MODELS` in `ml/models.py` maps `densenet`/`mobilenet` to the class and the dashboard filenames. Adding a model = add an entry there.

**Outputs of `train.py`** per model: `ml/weights/{name}_seed{n}.pth`, `ml/weights/{name}.pth` (copy of the best-val seed, served by the dashboard), and in `dashboard/public/data/`: `metrics.json` / `metrics_model2.json` (means, `*_std`, per-seed results, `selected_seed`, `config`) and `training_history.json` / `training_history_model2.json` (selected seed). `dashboard/src/app/page.tsx` reads these; keep their keys in sync.

**Inference path**: `dashboard/src/app/api/predict/route.ts` saves the upload to `dashboard/public/uploads/`, then runs `python3 ../ml/inference.py <file> --model <name>` via `execFile` (no shell) for both models in parallel. `inference.py` must print exactly one JSON line — `{class, confidence, probabilities: {Salmon, Trout}}` or `{error}`; extra stdout breaks `JSON.parse`. Run `npm run dev` from inside `dashboard/` since the route resolves `../ml` from `process.cwd()`.

**Explanation (NLP)**: after both predictions the route runs `ml/explain.py <file> --predictions '<json>'`, which prints `{summary, features}` — a Thai template-generated summary comparing the image's flesh colour (brightness, redness, saturation) with per-class train-split averages in `ml/feature_stats.json` (rebuild with `--build-stats` if the splits change). It describes the image, not CNN attention. If `GEMINI_API_KEY` is set (put it in `dashboard/.env.local`; Next passes it to the child process), Gemini rewrites that template summary into natural Thai (`source: "gemini"`); no key or any API error falls back to the template (`source: "template"`). `GEMINI_MODEL` overrides the default model. Uses `certifi` because python.org macOS builds lack a CA bundle. A failure there leaves `explanation: null` and the predictions still show.

## Gotchas

- Class index order is fixed: `0 = Salmon`, `1 = Trout` (`CLASSES` in `ml/config.py`).
- Seeds are set, but MPS is not bit-for-bit deterministic; that's why results are reported as mean ± SD.
- `.gitignore` has Python packaging rules; `lib/` is anchored to the root (`/lib/`) because the unanchored form previously hid `dashboard/src/lib/utils.ts`.
- `npm run lint` has pre-existing `no-explicit-any` errors in `page.tsx`; `npm run build` passes.
