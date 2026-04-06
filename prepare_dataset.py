"""
Dataset preparation — Week 7 pixel mask generation approach.

Primary method: SAM (Segment Anything Model) auto-mask generator.
  For each aerial image, SAM proposes candidate segments. Each candidate
  is filtered by measuring its IoU against the dataset's bounding-box
  annotations. Candidates that overlap a labelled building bbox above the
  configured threshold are merged into the final house mask.
  This is the technique demonstrated in Week 7.

Fallback (no SAM checkpoint): polygon annotations are rasterised directly
  using PIL when 'SAM_CHECKPOINT_PATH' is not set or the file is absent.

Usage:
    # With SAM (Week 7 method — preferred):
    #   1. Download sam_vit_h_4b8939.pth from Meta and put it in this folder.
    #   2. Set SAM_CHECKPOINT_PATH in your .env (or leave default).
    python prepare_dataset.py

Environment variables (via .env or shell):
    SAM_CHECKPOINT_PATH — path to SAM ViT-H weights  (default: sam_vit_h_4b8939.pth)
    IOU_THRESHOLD       — min IoU to accept a SAM mask (default: 0.3)
    DATA_OUTPUT_DIR     — where to write the dataset   (default: data/)
    MAX_SAMPLES         — cap the number of samples    (default: 500)
"""

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SAM_CHECKPOINT   = os.getenv("SAM_CHECKPOINT_PATH", "sam_vit_h_4b8939.pth")
IOU_THRESHOLD    = float(os.getenv("IOU_THRESHOLD", "0.3"))
SPLIT_RATIOS     = (0.70, 0.15, 0.15)          # train / val / test
OUTPUT_DIR       = Path(os.getenv("DATA_OUTPUT_DIR", "data"))
MAX_SAMPLES      = int(os.getenv("MAX_SAMPLES", "500"))
RANDOM_SEED      = 42


# ── SAM helpers (Week 7 method) ───────────────────────────────────────────────
def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Pixel-level IoU between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union        = np.logical_or(mask_a,  mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def bbox_to_binary_mask(bbox, image: Image.Image) -> np.ndarray:
    """Convert a [x, y, w, h] bounding box to a binary mask."""
    x, y, w, h = (int(v) for v in bbox)
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    mask[y : y + h, x : x + w] = 1
    return mask


def build_sam_house_mask(example, sam_masks) -> np.ndarray:
    """
    Week 7 mask generation:
    For every candidate segment SAM proposes, check its IoU against each
    labelled bounding box. If any overlap exceeds IOU_THRESHOLD, the segment
    is considered a building and merged into the combined mask.
    """
    image    = example["image"]
    combined = np.zeros((image.height, image.width), dtype=np.uint8)

    for sam_m in sam_masks:
        sam_seg = sam_m["segmentation"].astype(np.uint8)
        for bbox in example["objects"]["bbox"]:
            label_seg = bbox_to_binary_mask(bbox, image)
            if compute_iou(sam_seg, label_seg) > IOU_THRESHOLD:
                combined = np.logical_or(combined, sam_seg).astype(np.uint8)
                break   # one matching bbox is enough for this SAM segment

    return combined


# ── Polygon fallback (no SAM checkpoint) ─────────────────────────────────────
def poly_to_mask(image: Image.Image, segmentations: list) -> np.ndarray:
    """
    Rasterise polygon annotations into a binary pixel mask.
    Each entry in `segmentations` is a flat list [x1, y1, x2, y2, ...].
    """
    mask_img = Image.new("L", (image.width, image.height), 0)
    draw     = ImageDraw.Draw(mask_img)
    for poly in segmentations:
        if len(poly) < 6:
            continue
        coords = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
        draw.polygon(coords, fill=1)
    return np.array(mask_img, dtype=np.uint8)


def fallback_mask(example) -> np.ndarray:
    """Build a mask without SAM: polygons first, bounding boxes otherwise."""
    image = example["image"]
    objs  = example.get("objects", {})
    segs  = objs.get("segmentation", [])
    if segs:
        return poly_to_mask(image, segs)
    bboxes = objs.get("bbox", [])
    mask   = np.zeros((image.height, image.width), dtype=np.uint8)
    for bbox in bboxes:
        x, y, w, h = (int(v) for v in bbox)
        mask[y : y + h, x : x + w] = 1
    return mask


# ── I/O ───────────────────────────────────────────────────────────────────────
def save_pair(image: Image.Image, mask: np.ndarray, dest: Path, stem: str):
    dest.mkdir(parents=True, exist_ok=True)
    image.save(dest / f"{stem}.jpg")
    Image.fromarray(mask * 255).save(dest / f"{stem}_mask.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Decide which method to use
    use_sam = Path(SAM_CHECKPOINT).exists()
    if use_sam:
        print(f"[SAM] Checkpoint found: {SAM_CHECKPOINT}  (Week 7 method)")
    else:
        print(f"[WARNING] SAM checkpoint not found at '{SAM_CHECKPOINT}'.")
        print(f"          Falling back to polygon/bbox mask extraction.")
        print(f"          To use the Week 7 SAM method, download sam_vit_h_4b8939.pth")
        print(f"          from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")

    print("\n[1/4] Loading dataset from HuggingFace...")
    ds = load_dataset("keremberke/satellite-building-segmentation", name="full")

    all_examples = []
    for split_name in ds:
        for ex in ds[split_name]:
            all_examples.append(ex)
            if len(all_examples) >= MAX_SAMPLES:
                break
        if len(all_examples) >= MAX_SAMPLES:
            break
    random.shuffle(all_examples)
    print(f"   Collected {len(all_examples)} samples")

    if use_sam:
        print(f"[2/4] Loading SAM ({device})...")
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)
    else:
        mask_generator = None

    print("[3/4] Generating pixel masks...")
    pairs   = []
    skipped = 0

    for i, ex in enumerate(all_examples):
        print(f"   [{i + 1}/{len(all_examples)}] processing...", end="\r")
        image = ex["image"].convert("RGB")

        if use_sam:
            img_np = np.array(image)
            sam_masks = mask_generator.generate(img_np)
            mask = build_sam_house_mask(ex, sam_masks)
        else:
            mask = fallback_mask(ex)

        if mask.sum() == 0:
            skipped += 1
            continue
        pairs.append((image, mask))

    print(f"\n   Kept {len(pairs)} samples  ({skipped} skipped — no annotation)")

    print("[4/4] Splitting and saving...")
    n       = len(pairs)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val   = int(n * SPLIT_RATIOS[1])
    splits  = {
        "train": pairs[:n_train],
        "val"  : pairs[n_train : n_train + n_val],
        "test" : pairs[n_train + n_val :],
    }

    manifest = {}
    for split_name, split_pairs in splits.items():
        manifest[split_name] = []
        for j, (img, msk) in enumerate(split_pairs):
            stem = f"{split_name}_{j:04d}"
            save_pair(img, msk, OUTPUT_DIR / split_name, stem)
            manifest[split_name].append(stem)
        print(f"   {split_name}: {len(split_pairs)} samples -> {OUTPUT_DIR / split_name}")

    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! Dataset saved to: {OUTPUT_DIR}")
    print(f"  train: {len(splits['train'])}  val: {len(splits['val'])}  test: {len(splits['test'])}")


if __name__ == "__main__":
    main()
