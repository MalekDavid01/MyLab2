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
import requests
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Load secrets / config from .env (never hard-coded)
load_dotenv()

from flask import Flask, request, jsonify

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints/best")
IMG_SIZE       = int(os.getenv("IMG_SIZE", "512"))
PORT           = int(os.getenv("PORT", "5000"))
# Optional: HuggingFace token for private model repos
HF_TOKEN       = os.getenv("HF_TOKEN", None)

# ── Lazy-loaded model state ───────────────────────────────────────────────────
processor        = None
model            = None
model_load_error = None
_device          = None   # resolved on first model load


def _get_device() -> str:
    """Resolve compute device; deferred so torch is not imported at module load."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_model_if_needed() -> bool:
    """Load model artifacts only when required, so import/tests don't hard-fail."""
    global processor, model, model_load_error, _device

    if processor is not None and model is not None:
        return True

    try:
        import torch
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model from '{CHECKPOINT_DIR}' on {_device}...")
        processor = SegformerImageProcessor.from_pretrained(CHECKPOINT_DIR, token=HF_TOKEN)
        model     = SegformerForSemanticSegmentation.from_pretrained(CHECKPOINT_DIR, token=HF_TOKEN)
        model.to(_device).eval()
        model_load_error = None
        print("Model ready.")
        return True
    except Exception as exc:
        model_load_error = str(exc)
        processor = None
        model     = None
        print(f"Model load failed: {model_load_error}")
        return False


app = Flask(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def predict_mask(image: Image.Image):
    """Run inference; returns binary mask (H×W numpy) and per-image metrics."""
    import torch
    import torch.nn.functional as F

    orig_w, orig_h = image.size
    encoded = processor(
        images=image,
        return_tensors="pt",
        size={"height": IMG_SIZE, "width": IMG_SIZE},
    )
    pixel_values = encoded["pixel_values"].to(_device)

    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits   # (1, C, H/4, W/4)

    # Upsample back to original resolution
    upsampled = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    pred_mask = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    house_pixels = int(pred_mask.sum())
    total_pixels = orig_h * orig_w
    coverage_pct = round(100 * house_pixels / total_pixels, 2)

    return pred_mask, house_pixels, coverage_pct


def mask_to_base64(mask: np.ndarray) -> str:
    """Encode binary mask (0/1) as a base64 PNG string."""
    img = Image.fromarray(mask * 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    model_ready = processor is not None and model is not None
    device      = _device if _device else _get_device()
    response = {
        "status"     : "ok",
        "device"     : device,
        "model"      : CHECKPOINT_DIR,
        "model_ready": model_ready,
    }
    if model_load_error:
        response["model_error"] = model_load_error
    return jsonify(response)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not load_model_if_needed():
            return jsonify({
                "error"  : "Model is not available. Train/download checkpoint first.",
                "details": model_load_error,
            }), 503

        data = request.get_json(silent=True)   # returns None on parse error
        if not data or "image_url" not in data:
            return jsonify({"error": "Missing 'image_url' in request body"}), 400

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

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Could not fetch image: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model_if_needed()
    from waitress import serve
    print(f"Starting server on port {PORT}...")
    serve(app, host="0.0.0.0", port=PORT)
