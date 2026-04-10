# Lab 2 - Aerial House Segmentation Service
Github repo found here: https://github.com/MalekDavid01/MyLab2

Extends Lab 1 (ResNet-50 image classifier) with:

- **Secrets injection** via `.env` / `python-dotenv`
- **CI/CD** pipeline (GitHub Actions) -- tests -> Docker build -> Docker Hub push
- **Dataset preparation** -- Week 7 SAM pixel mask generation on aerial images
- **Segmentation model** -- fine-tuned SegFormer (`nvidia/mit-b0`) replacing Lab 1's ResNet-50
- **Metrics** -- IoU and Dice score tracked per epoch and reported on the test set

Docker Hub image: `dvdmalek/house-segmentation-service`

---

## Project structure

```
|-- app.py                        # Flask/Waitress inference API
|-- train.py                      # SegFormer fine-tuning loop (IoU + Dice tracking)
|-- evaluate.py                   # Test-set evaluation + prediction visualisations
|-- prepare_dataset.py            # Week 7 SAM mask generation + dataset builder
|-- test_api.py                   # 10 automated API endpoint tests
|-- requirements.txt
|-- Dockerfile
|-- .env.example                  # Copy to .env and fill in your values
|-- .github/workflows/ci-cd.yml   # GitHub Actions CI/CD pipeline
|-- SCREENSHOTS/                  # Contains all screenshots required for submission
|-- eval_outputs/                 # generated metrics and prediction figures
|-- checkpoints/best/             # trained model artifacts needed for local inference
|-- Lab2Report.pdf                # final report to submit
```

---

## 1 - Secrets injection

All configuration and sensitive values live in `.env` (never committed to git).

```bash
cp .env.example .env
# edit .env with your values
```

Every script calls `load_dotenv()` at startup. The `.gitignore` excludes `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `CHECKPOINT_DIR` | `checkpoints/best` | Where the trained model is saved/loaded |
| `PORT` | `5000` | Flask server port |
| `SEGFORMER_MODEL` | `nvidia/mit-b0` | Base model for fine-tuning |
| `EPOCHS` | `5` | Training epochs |
| `BATCH_SIZE` | `2` | Batch size |
| `LEARNING_RATE` | `6e-5` | AdamW learning rate |
| `IMG_SIZE` | `256` | Input resolution |
| `SAM_CHECKPOINT_PATH` | `sam_vit_h_4b8939.pth` | Path to SAM ViT-H weights |
| `IOU_THRESHOLD` | `0.3` | Min IoU to accept a SAM segment as a building |
| `DATA_OUTPUT_DIR` | `data` | Dataset output directory |
| `MAX_SAMPLES` | `100` | Number of samples to use |

---

## 2 - CI/CD pipeline

Workflow: `.github/workflows/ci-cd.yml` — triggers on every push to `main`.

**Stage 1 - Test**
- Installs CPU-only PyTorch + `requirements.txt`
- Runs `pytest test_api.py -v --cov=app` (10 tests, no GPU needed)

**Stage 2 - Build & Push** (only on `main` push, after tests pass)
- Builds multi-stage Docker image
- Pushes to Docker Hub: `dvdmalek/house-segmentation-service:latest`

Required GitHub repository secrets:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

---

## 3 - Dataset preparation (Week 7 method)

```bash
python prepare_dataset.py
```

Uses the `make_mask()` approach from the Week 7 notebook:
1. Loads `keremberke/satellite-building-segmentation` from HuggingFace.
2. For each image, converts bounding box annotations to binary pixel masks using `make_mask()`.
3. When SAM checkpoint is present, SAM candidate segments are filtered by IoU against the bbox masks (threshold 0.3) for higher-quality pixel-level masks.
4. Splits 70/15/15 into `data/train`, `data/val`, `data/test`.
5. Writes `data/manifest.json`.

---

## 4 - Training

```bash
python train.py
```

Fine-tunes `nvidia/mit-b0` (SegFormer) with a 2-class segmentation head (background / house).
Optimizer: AdamW + cosine annealing. Best checkpoint saved to `checkpoints/best/`.

Actual training results (5 epochs, CPU, 63 training samples):

| Epoch | Train Loss | Val Loss | Val IoU | Val Dice |
|-------|-----------|---------|--------|---------|
| 1 | 0.6546 | 0.6133 | 0.4430 | 0.6023 |
| 2 | 0.5027 | 0.4541 | 0.5370 | 0.6895 |
| 3 | 0.4190 | 0.4630 | 0.5435 | 0.6946 |
| 4 | 0.4086 | 0.4121 | 0.5614 | 0.7099 |
| 5 | 0.3780 | 0.3859 | 0.5634 | 0.7109 |

Best: epoch 5 -- Val IoU = 0.5634, Val Dice = 0.7109

---

## 5 - Evaluation

```bash
python evaluate.py
```

Runs the best checkpoint on `data/test/` (15 samples).

**Test set results:**
- Mean IoU: **0.4589** (+/- 0.2212)
- Mean Dice: **0.5934** (+/- 0.2408)

Outputs saved to `eval_outputs/`:
- `predictions_grid.png` -- aerial image | ground truth mask | predicted mask
- `training_curves.png` -- loss + IoU/Dice over epochs
- `metrics.json` -- full numeric results

**Training issues and how they were addressed:**

1. **Class imbalance** -- aerial images have far more background than building pixels. Samples with zero house pixels were discarded during dataset preparation. A class-weighted loss would further help.

2. **Small dataset** -- 63 training samples is very few. Cosine annealing and best-val-IoU checkpointing prevent overfitting. Train/val loss both decreased across epochs confirming no overfitting occurred.

3. **SAM availability** -- SAM requires a 2.4 GB checkpoint download. The script falls back to the Week 7 `make_mask()` bbox approach when the SAM file is not present, ensuring the pipeline always runs.

---

## 6 - Inference API

### Run locally

```bash
python app.py
```

### Pull and run from Docker Hub

```bash
# Pull the pre-built image
docker pull dvdmalek/house-segmentation-service:latest

# Run -- mount your trained checkpoint directory
docker run -p 5000:5000 \
  --env-file .env \
  -v $(pwd)/checkpoints:/app/checkpoints \
  dvdmalek/house-segmentation-service:latest
```

The model checkpoint is not baked into the image (it changes with each training run).
Mount the `checkpoints/` folder produced by `python train.py` at runtime.
`GET /health` reports `"model_ready": true` once the model is loaded.

### Endpoints

**GET /health**
```json
{"status": "ok", "device": "cpu", "model": "checkpoints/best", "model_ready": true}
```

**POST /predict**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/aerial.jpg"}'
```
```json
{
  "input": "https://example.com/aerial.jpg",
  "image_size": {"width": 512, "height": 512},
  "house_pixels": 12540,
  "coverage_percent": 4.78,
  "mask_png_base64": "..."
}
```

Note: `/predict` does not return IoU or Dice -- those require a ground-truth mask.
Run `python evaluate.py` to get segmentation metrics on the labelled test set.

### Quick prediction smoke test

Use this after starting the API locally or inside Docker to confirm inference works. The command is the same for both when the service is exposed on port 5000:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body (@{ image_url = "https://images.pexels.com/photos/1029613/pexels-photo-1029613.jpeg" } | ConvertTo-Json)
```

Expected response fields:
- `input`
- `image_size`
- `house_pixels`
- `coverage_percent`
- `mask_png_base64`

If `/health` shows `model_ready: true` and `/predict` returns those fields, the API is working correctly.

If you prefer to check the health endpoint first, use this same command for both local API and Docker:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/health -Method Get
```

---

## 7 - Running the tests

```bash
# Activate venv first
venv\Scripts\activate

pytest test_api.py -v
```

10 tests covering `/health` and `/predict` -- all pass without GPU or internet.
