"""
Dataset preparation — Week 7 pixel mask generation approach.

Follows the technique from the Week 7 notebook (segment_houses_smartly):
  1. Load the satellite-building-segmentation dataset (bounding box labels).
  2. For each image, use SAM's automatic mask generator to propose segments.
  3. For each SAM segment, compute IoU against every labelled bounding box.
  4. Accept segments with IoU > threshold and merge into a binary house mask.

Fallback (no SAM checkpoint available): build the mask directly from the
bounding box labels using the same make_mask() logic as the Week 7 notebook.

Usage:
    python prepare_dataset.py

Environment variables (via .env):
    SAM_CHECKPOINT_PATH — path to SAM ViT-H weights  (default: sam_vit_h_4b8939.pth)
    IOU_THRESHOLD       — min IoU to accept a SAM mask (default: 0.3)
    DATA_OUTPUT_DIR     — where to write the dataset   (default: data/)
    MAX_SAMPLES         — cap the number of samples    (default: 500)
"""

import os
import json
import random
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT_PATH", "sam_vit_h_4b8939.pth")
IOU_THRESHOLD  = float(os.getenv("IOU_THRESHOLD", "0.3"))
SPLIT_RATIOS   = (0.70, 0.15, 0.15)
OUTPUT_DIR     = Path(os.getenv("DATA_OUTPUT_DIR", "data"))
MAX_SAMPLES    = int(os.getenv("MAX_SAMPLES", "500"))
RANDOM_SEED    = 42


# ── Mask helpers (exact Week 7 logic) ────────────────────────────────────────
def make_mask(labelled_bbox, image: Image.Image) -> np.ndarray:
    """
    Convert a bounding box [x_min, y_min, width, height] to a binary mask.
    Matches the make_mask() function from the Week 7 notebook exactly.
    """
    x_min, y_min, width, height = (int(v) for v in labelled_bbox)
    mask_instance = np.zeros((image.width, image.height))
    last_x = x_min + width
    last_y = y_min + height
    mask_instance[x_min:last_x, y_min:last_y] = np.ones((width, height))
    return mask_instance.T


def build_sam_house_mask(example, sam_masks, image: Image.Image) -> np.ndarray:
    """
    Week 7 method: for each SAM segment, check IoU against labelled bboxes.
    Merge all accepted segments into one combined binary mask.
    """
    combined = np.zeros((image.height, image.width), dtype=np.uint8)
    for sam_m in sam_masks:
        sam_seg = sam_m["segmentation"].astype(int)
        for bbox in example["objects"]["bbox"]:
            label_seg = make_mask(bbox, image)
            intersection = np.sum(np.logical_and(sam_seg, label_seg))
            union        = np.sum(np.logical_or(sam_seg, label_seg))
            iou = intersection / union if union > 0 else 0.0
            if iou > IOU_THRESHOLD:
                combined = np.logical_or(combined, sam_seg).astype(np.uint8)
                break
    return combined


def build_bbox_house_mask(example, image: Image.Image) -> np.ndarray:
    """
    Fallback (no SAM): union of all bounding-box masks using the same
    make_mask() logic as Week 7.
    """
    combined = np.zeros((image.height, image.width), dtype=np.uint8)
    for bbox in example["objects"]["bbox"]:
        m = make_mask(bbox, image)
        combined = np.logical_or(combined, m).astype(np.uint8)
    return combined


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

    use_sam = Path(SAM_CHECKPOINT).exists()
    if use_sam:
        print(f"[SAM] Checkpoint found: {SAM_CHECKPOINT}  (Week 7 method)")
    else:
        print(f"[WARNING] SAM checkpoint not found at '{SAM_CHECKPOINT}'.")
        print(f"          Falling back to bounding-box mask extraction (Week 7 make_mask logic).")
        print(f"          To use SAM, download sam_vit_h_4b8939.pth from:")
        print(f"          https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")

    print("\n[1/4] Loading dataset from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset(
        "keremberke/satellite-building-segmentation",
        name="full",
        trust_remote_code=True,
    )

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
            sam_masks = mask_generator.generate(np.array(image))
            mask = build_sam_house_mask(ex, sam_masks, image)
        else:
            mask = build_bbox_house_mask(ex, image)

        if mask.sum() == 0:
            skipped += 1
            continue
        pairs.append((image, mask))

    print(f"\n   Kept {len(pairs)} samples  ({skipped} skipped — no buildings)")

    print("[4/4] Splitting and saving...")
    n       = len(pairs)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val   = int(n * SPLIT_RATIOS[1])
    splits  = {
        "train": pairs[:n_train],
        "val"  : pairs[n_train : n_train + n_val],
        "test" : pairs[n_train + n_val :],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure each run writes a clean dataset and does not leave stale files.
    for split_name in splits:
        split_dir = OUTPUT_DIR / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)

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
