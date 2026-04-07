"""
Evaluation script.
Loads the best checkpoint, runs the test set, reports IoU + Dice,
and saves a visualization grid: image | ground truth | prediction.
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

load_dotenv()

DATA_DIR        = Path(os.getenv("DATA_OUTPUT_DIR", "data"))
CHECKPOINT_DIR  = Path(os.getenv("CHECKPOINT_DIR", "checkpoints")) / "best"
OUTPUT_DIR      = Path(os.getenv("EVAL_OUTPUT_DIR", "eval_outputs"))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE", "8"))
IMG_SIZE        = int(os.getenv("IMG_SIZE", "512"))
VIZ_SAMPLES     = int(os.getenv("VIZ_SAMPLES", "8"))
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


# Re-use dataset class from train.py
from train import HouseSegDataset, compute_metrics


def evaluate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processor = SegformerImageProcessor.from_pretrained(str(CHECKPOINT_DIR))
    model     = SegformerForSemanticSegmentation.from_pretrained(str(CHECKPOINT_DIR)).to(DEVICE)
    model.eval()

    test_ds = HouseSegDataset("test", processor)
    _nw = 0 if os.name == "nt" else 4
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=_nw)

    iou_list, dice_list = [], []
    viz_data = []   # (image_path, gt_mask, pred_mask)

    with open(DATA_DIR / "manifest.json") as f:
        test_stems = json.load(f)["test"]

    sample_idx = 0
    with torch.no_grad():
        for pixel_values, labels in test_dl:
            pixel_values = pixel_values.to(DEVICE)
            labels       = labels.to(DEVICE)
            outputs      = model(pixel_values=pixel_values)

            logits    = outputs.logits
            upsampled = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            preds     = upsampled.argmax(dim=1)

            for i in range(len(preds)):
                iou, dice = compute_metrics(preds[i:i+1], labels[i:i+1])
                iou_list.append(iou)
                dice_list.append(dice)

                if len(viz_data) < VIZ_SAMPLES:
                    stem     = test_stems[sample_idx]
                    img_path = DATA_DIR / "test" / f"{stem}.jpg"
                    viz_data.append((img_path,
                                     labels[i].cpu().numpy(),
                                     preds[i].cpu().numpy()))
                sample_idx += 1

    mean_iou  = np.mean(iou_list)
    mean_dice = np.mean(dice_list)
    std_iou   = np.std(iou_list)
    std_dice  = np.std(dice_list)

    results = dict(
        mean_iou =round(float(mean_iou),  4),
        std_iou  =round(float(std_iou),   4),
        mean_dice=round(float(mean_dice), 4),
        std_dice =round(float(std_dice),  4),
        n_samples=len(iou_list),
    )
    print("\n-- Test Set Results --")
    for k, v in results.items():
        print(f"  {k:<12}: {v}")
    print("---------------------\n")

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Visualization grid ────────────────────────────────────────────────
    n   = len(viz_data)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["Aerial Image", "Ground Truth Mask", "Predicted Mask"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=13, fontweight="bold", pad=10)

    for row, (img_path, gt, pred) in enumerate(viz_data):
        img = np.array(Image.open(img_path).convert("RGB"))
        axes[row][0].imshow(img)
        axes[row][1].imshow(gt,   cmap="gray", vmin=0, vmax=1)
        axes[row][2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        iou_val = iou_list[row]
        axes[row][2].set_xlabel(f"IoU={iou_val:.3f}", fontsize=10)
        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle(f"House Segmentation — Test Set  (mean IoU={mean_iou:.3f}, Dice={mean_dice:.3f})",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    viz_path = OUTPUT_DIR / "predictions_grid.png"
    plt.savefig(viz_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Visualization saved -> {viz_path}")

    # ── Loss curves (from training history) ───────────────────────────────
    hist_path = Path(os.getenv("CHECKPOINT_DIR", "checkpoints")) / "history.json"
    if hist_path.exists():
        with open(hist_path) as f:
            history = json.load(f)
        epochs     = [r["epoch"]      for r in history]
        train_loss = [r["train_loss"] for r in history]
        val_loss   = [r["val_loss"]   for r in history]
        iou_curve  = [r["iou"]        for r in history]
        dice_curve = [r["dice"]       for r in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, train_loss, label="Train Loss", color="#e74c3c")
        ax1.plot(epochs, val_loss,   label="Val Loss",   color="#3498db")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss"); ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(epochs, iou_curve,  label="Val IoU",  color="#2ecc71")
        ax2.plot(epochs, dice_curve, label="Val Dice", color="#9b59b6")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
        ax2.set_title("IoU & Dice Score (Validation)"); ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        curve_path = OUTPUT_DIR / "training_curves.png"
        plt.savefig(curve_path, dpi=120)
        plt.close()
        print(f"Training curves saved -> {curve_path}")


if __name__ == "__main__":
    evaluate()
