"""
Flask / Waitress inference server.
Serves the fine-tuned SegFormer house-segmentation model.

Endpoints:
  GET  /health   → liveness check
  POST /predict  → { "image_url": "..." }  → mask + metrics JSON
"""

import os
import io
import base64
from pathlib import Path
import requests
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Load secrets / config from .env (never hard-coded)
load_dotenv()

from flask import Flask, request, jsonify

# Config
# Directory containing SegFormer model files (config, weights, processor files).
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints/best")
# Inference resize size; inputs are resized before being passed to the model.
IMG_SIZE       = int(os.getenv("IMG_SIZE", "512"))
# API port exposed by Flask/Waitress.
PORT           = int(os.getenv("PORT", "5000"))
# Optional: HuggingFace token for private model repos
HF_TOKEN       = os.getenv("HF_TOKEN", None)

# Lazy-loaded model state
processor        = None
model            = None
model_load_error = None
_device          = None   # resolved on first model load
_model_dir       = None   # resolved checkpoint path in use


def _get_device() -> str:
    """Resolve compute device; deferred so torch is not imported at module load."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_model_if_needed() -> bool:
    """Load model artifacts only when required, so import/tests don't hard-fail."""
    global processor, model, model_load_error, _device, _model_dir

    # If model and processor are already initialized, skip reloading.
    if processor is not None and model is not None:
        return True

    try:
        import torch
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
        # Pick GPU if available, otherwise default to CPU.
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        configured_dir = Path(CHECKPOINT_DIR)
        # Resolve path for both current and legacy checkpoint layouts.
        if (configured_dir / "config.json").exists():
            _model_dir = configured_dir
        elif (configured_dir / "best" / "config.json").exists():
            # Backward compatibility for older runs that saved under <CHECKPOINT_DIR>/best.
            _model_dir = configured_dir / "best"
        else:
            _model_dir = configured_dir

        print(f"Loading model from '{_model_dir}' on {_device}...")
        # Load preprocessor and model from the selected checkpoint directory.
        processor = SegformerImageProcessor.from_pretrained(str(_model_dir), token=HF_TOKEN)
        model     = SegformerForSemanticSegmentation.from_pretrained(str(_model_dir), token=HF_TOKEN)
        model.to(_device).eval()
        model_load_error = None
        print("Model ready.")
        return True
    except Exception as exc:
        # Save error details so /health and /predict can return useful diagnostics.
        model_load_error = str(exc)
        processor = None
        model     = None
        print(f"Model load failed: {model_load_error}")
        return False


app = Flask(__name__)


# Helpers
def fetch_image(url: str) -> Image.Image:
    # Download the input image from URL and convert to RGB for model inference.
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def predict_mask(image: Image.Image):
    """Run inference; returns binary mask (H×W numpy) and per-image metrics."""
    import torch
    import torch.nn.functional as F

    # Keep original size so we can upsample model output back to the source resolution.
    orig_w, orig_h = image.size
    # Convert image to model tensor format.
    encoded = processor(
        images=image,
        return_tensors="pt",
        size={"height": IMG_SIZE, "width": IMG_SIZE},
    )
    pixel_values = encoded["pixel_values"].to(_device)

    # Run inference without gradients for speed and reduced memory usage.
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits   # (1, C, H/4, W/4)

    # Upsample back to original resolution
    upsampled = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    pred_mask = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Class 1 represents house/building pixels in this binary setup.
    house_pixels = int(pred_mask.sum())
    total_pixels = orig_h * orig_w
    coverage_pct = round(100 * house_pixels / total_pixels, 2)

    return pred_mask, house_pixels, coverage_pct


def mask_to_base64(mask: np.ndarray) -> str:
    """Encode binary mask (0/1) as a base64 PNG string."""
    # Convert mask to visible grayscale PNG: 0 -> black, 1 -> white.
    img = Image.fromarray(mask * 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Routes
@app.route("/health", methods=["GET"])
def health():
    # Report readiness and runtime metadata for local and Docker health checks.
    model_ready = processor is not None and model is not None
    device      = _device if _device else _get_device()
    response = {
        "status"     : "ok",
        "device"     : device,
        "model"      : str(_model_dir) if _model_dir else CHECKPOINT_DIR,
        "model_ready": model_ready,
    }
    # Include load error when model initialization failed.
    if model_load_error:
        response["model_error"] = model_load_error
    return jsonify(response)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load model lazily so unit tests and import-time checks are lightweight.
        if not load_model_if_needed():
            return jsonify({
                "error"  : "Model is not available. Train/download checkpoint first.",
                "details": model_load_error,
            }), 503

        # Validate request payload before trying to fetch or infer.
        data = request.get_json(silent=True)   # returns None on parse error
        if not data or "image_url" not in data:
            return jsonify({"error": "Missing 'image_url' in request body"}), 400

        # Fetch remote image and run segmentation prediction.
        # Read required URL input from request body.
        image_url = data["image_url"]
        image     = fetch_image(image_url)
        mask, house_pixels, coverage_pct = predict_mask(image)

        return jsonify({
            "input"           : image_url,
            "image_size"      : {"width": image.width, "height": image.height},
            "house_pixels"    : house_pixels,
            "coverage_percent": coverage_pct,
            "mask_png_base64" : mask_to_base64(mask),
        }), 200

    # Network and URL fetching errors are returned as 400 (bad input URL or network issue).
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Could not fetch image: {e}"}), 400
    # Any unexpected internal error is returned as 500.
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Entry point
if __name__ == "__main__":
    # Preload model at startup so runtime requests are faster.
    load_model_if_needed()
    from waitress import serve
    print(f"Starting server on port {PORT}...")
    serve(app, host="0.0.0.0", port=PORT)
