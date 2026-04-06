# Lab 2 - Aerial House Segmentation Service

Extends Lab 1 (ResNet-50 image classifier) with:

- **Secrets injection** via `.env` / `python-dotenv`
- **CI/CD** pipeline (GitHub Actions) -- tests -> Docker build -> Docker Hub push
- **Dataset preparation** -- SAM pixel mask generation (Week 7 method) on aerial images
- **Segmentation model** -- fine-tuned SegFormer (`nvidia/mit-b2`) for binary house segmentation
- **Metrics** -- IoU and Dice score reported per epoch and on the held-out test set

---

## Project structure

```
Lab2/
|-- app.py                  # Flask/Waitress inference server
|-- train.py                # SegFormer fine-tuning loop
|-- evaluate.py             # Test-set evaluation + visualisations
|-- prepare_dataset.py      # SAM pixel mask generation (Week 7) + dataset builder
|-- test_api.py             # Pytest unit tests for the API
|-- requirements.txt
|-- Dockerfile
|-- .env.example            # Copy to .env and fill in your values
```

---

## 1 - Secrets injection

All sensitive values and tuneable parameters live in `.env` (never committed).

```bash
cp .env.example .env
# edit .env with your values
```

`app.py`, `train.py`, `evaluate.py`, and `prepare_dataset.py` all call
`load_dotenv()` at startup, so secrets are loaded from `.env` automatically.
The `.gitignore` explicitly excludes `.env`.

Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | (empty) | HuggingFace token for private repos |
| `CHECKPOINT_DIR` | `checkpoints/best` | Where the trained model is saved/loaded |
| `PORT` | `5000` | Flask server port |
| `EPOCHS` | `10` | Training epochs |
| `BATCH_SIZE` | `8` | Batch size |
| `LEARNING_RATE` | `6e-5` | AdamW learning rate |
| `IMG_SIZE` | `512` | Input resolution |
| `SAM_CHECKPOINT_PATH` | `sam_vit_h_4b8939.pth` | Path to SAM ViT-H weights |
| `IOU_THRESHOLD` | `0.3` | Min IoU to accept a SAM mask as a building |
| `DATA_OUTPUT_DIR` | `data` | Dataset output directory |
| `MAX_SAMPLES` | `500` | Max samples to download |

---

## 2 - CI/CD pipeline

The GitHub Actions workflow at `.github/workflows/ci-cd.yml` runs on every push to `main` or `develop`:

**Stage 1 - Test** (all branches)
- Sets up Python 3.11
- Installs CPU-only PyTorch (avoids the 2 GB CUDA download in CI)
- Installs `requirements.txt`
- Runs `pytest test_api.py -v --cov=app`

**Stage 2 - Build & Push** (only `main` push, after tests pass)
- Logs in to Docker Hub using `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` repository secrets
- Builds a multi-stage Docker image
- Pushes tagged `latest` + short-SHA image to `dvdmalek/house-segmentation-service`

To enable the Docker push, add these secrets in your GitHub repository
Settings -> Secrets and variables -> Actions:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

---

## 3 - Dataset preparation (Week 7: SAM pixel mask generation)

```bash
# Step 1: download the SAM ViT-H checkpoint (~2.4 GB, one-time)
# https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# Place it in this folder or set SAM_CHECKPOINT_PATH in .env

# Step 2: run the script
python prepare_dataset.py
```

**How it works (Week 7 method):**

1. Loads the `keremberke/satellite-building-segmentation` aerial image dataset from HuggingFace.
2. For each image, SAM's `SamAutomaticMaskGenerator` proposes many candidate segments.
3. Each candidate segment's IoU is computed against every labelled building bounding box.
4. Candidates whose IoU exceeds `IOU_THRESHOLD` are accepted and merged into a single binary pixel mask.
5. Images with no accepted segments are skipped.
6. The result is split 70 / 15 / 15 into `data/train`, `data/val`, `data/test`.
7. `data/manifest.json` is written listing the stem names per split.

**Fallback (no SAM checkpoint):** If `sam_vit_h_4b8939.pth` is not found, the script
falls back to rasterising the polygon annotations bundled with the dataset directly.

Output layout:
```
data/
|-- manifest.json
|-- train/
|   |-- train_0000.jpg
|   |-- train_0000_mask.png
|   `-- ...
|-- val/
`-- test/
```

---

## 4 - Training

```bash
python train.py
```

- Fine-tunes `nvidia/mit-b2` (SegFormer) with a binary segmentation head.
- Optimiser: AdamW + cosine annealing scheduler.
- Tracks **IoU** and **Dice score** on the validation set each epoch.
- Saves the best checkpoint to `checkpoints/best/`.
- Writes `checkpoints/history.json` with the full loss/metric curve.

Sample epoch output:
```
Epoch 01/10 | train_loss=0.6821  val_loss=0.5103  IoU=0.3412  Dice=0.5089
   New best IoU=0.3412 -- checkpoint saved
```

---

## 5 - Evaluation

```bash
python evaluate.py
```

- Loads `checkpoints/best/` and runs inference on `data/test/`.
- Prints and saves per-sample IoU / Dice to `eval_outputs/metrics.json`.
- Saves a visualisation grid (`eval_outputs/predictions_grid.png`):
  aerial image | ground-truth mask | predicted mask
- Saves training curves (`eval_outputs/training_curves.png`):
  loss curves + IoU / Dice over epochs.

### Training issues encountered and how they were addressed

**1. Class imbalance (background >> houses)**

Aerial images contain far more background pixels than building pixels. This caused
the model to predict mostly background, resulting in low IoU in early epochs. The
SegFormer model is trained with the built-in cross-entropy loss from HuggingFace
Transformers, which computes per-pixel loss uniformly. To partially address the
imbalance, samples with zero house pixels are filtered out during dataset
preparation (`prepare_dataset.py`, the `mask.sum() == 0` check). A class-weighted
loss or focal loss would further improve this and is noted as future work.

**2. SAM mask quality vs. bounding box labels**

SAM sometimes over-segments (splitting one building into several masks) or
under-segments (grouping multiple buildings into one). The IoU filter
(`IOU_THRESHOLD=0.3`) was tuned to balance recall vs. noise. Raising it to 0.5
produced cleaner but fewer masks; lowering it to 0.1 included too many false
positives (roads, trees). 0.3 was the best compromise found during dataset
preparation.

**3. Potential overfitting with small datasets**

With only 500 samples, SegFormer can overfit after ~15 epochs. The cosine
annealing scheduler gradually reduces the learning rate, which slows down
overfitting. Monitoring the val_loss vs. train_loss gap in
`eval_outputs/training_curves.png` is the key diagnostic. The checkpoint saved
is always the one with the best **validation** IoU, not the final epoch, which
ensures generalisation is prioritised over training-set performance.

---

## 6 - Inference API

### Run locally

```bash
python app.py
```

### Run with Docker

The Docker image does **not** bake in the model checkpoint (it is large and
excluded from version control). You must mount the trained checkpoint directory
at runtime. After running `python train.py`, your checkpoint lives in
`checkpoints/best/`. Mount it like this:

```bash
docker build -t house-seg .

# $(pwd)/checkpoints must contain the best/ subfolder produced by train.py
docker run -p 5000:5000 \
  --env-file .env \
  -v $(pwd)/checkpoints:/app/checkpoints \
  house-seg
```

If the checkpoint is not mounted, `/predict` returns HTTP 503 with a clear
error message. You can verify the model is loaded with `GET /health`
(`"model_ready": true`).

### Endpoints

**GET /health**

```json
{
  "status": "ok",
  "device": "cpu",
  "model": "checkpoints/best",
  "model_ready": true
}
```

**POST /predict**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/aerial.jpg"}'
```

> **Note on metrics:** `/predict` does not return IoU or Dice score because
> those require a ground-truth mask, which is not available at inference time.
> Segmentation quality metrics are reported by `python evaluate.py`, which runs
> the model on the labelled test set and writes `eval_outputs/metrics.json`.

```json
{
  "input": "https://example.com/aerial.jpg",
  "image_size": {"width": 512, "height": 512},
  "house_pixels": 12540,
  "coverage_percent": 4.78,
  "mask_png_base64": "..."
}
```

---

## 7 - Running the tests

```bash
pytest test_api.py -v
```

All 10 tests mock out model and HTTP calls -- no GPU or internet required.
