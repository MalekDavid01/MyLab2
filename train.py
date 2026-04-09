"""
Training script for aerial house segmentation.
Fine-tunes SegFormer (nvidia/mit-b2) on the prepared dataset.
Tracks IoU and Dice score per epoch, saves best checkpoint.
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch.nn.functional as F

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(os.getenv("DATA_OUTPUT_DIR", "data"))
CHECKPOINT_DIR  = Path(os.getenv("CHECKPOINT_DIR", "checkpoints/best"))
MODEL_NAME      = os.getenv("SEGFORMER_MODEL", "nvidia/mit-b2")
EPOCHS          = int(os.getenv("EPOCHS", "10"))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE", "8"))
LR              = float(os.getenv("LEARNING_RATE", "6e-5"))
IMG_SIZE        = int(os.getenv("IMG_SIZE", "512"))
_default_workers = "0" if os.name == "nt" else "4"   # 0 on Windows (fork limitations)
NUM_WORKERS     = int(os.getenv("NUM_WORKERS", _default_workers))
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
NUM_LABELS      = 2   # background=0, house=1


# ── Dataset ───────────────────────────────────────────────────────────────────
class HouseSegDataset(Dataset):
    def __init__(self, split: str, processor: SegformerImageProcessor):
        self.processor = processor
        with open(DATA_DIR / "manifest.json") as f:
            stems = json.load(f)[split]
        self.pairs = [
            (DATA_DIR / split / f"{s}.jpg", DATA_DIR / split / f"{s}_mask.png")
            for s in stems
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask  = np.array(Image.open(msk_path).convert("L"))
        mask  = (mask > 127).astype(np.int64)   # binary 0/1

        encoded = self.processor(
            images=image,
            return_tensors="pt",
            size={"height": IMG_SIZE, "width": IMG_SIZE},
        )
        pixel_values = encoded["pixel_values"].squeeze(0)

        # Resize mask to match SegFormer's downsampled label size (÷4)
        label_size = IMG_SIZE // 4
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mask_resized = F.interpolate(mask_tensor, size=(label_size, label_size), mode="nearest")
        label = mask_resized.squeeze().long()

        return pixel_values, label


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(preds: torch.Tensor, labels: torch.Tensor):
    """
    preds  : (N, H, W) int64 — predicted class per pixel
    labels : (N, H, W) int64 — ground truth
    Returns mean IoU and Dice for the house class (class=1).
    """
    pred_flat  = (preds  == 1).float().view(-1)
    label_flat = (labels == 1).float().view(-1)

    intersection = (pred_flat * label_flat).sum()
    union        = pred_flat.sum() + label_flat.sum() - intersection

    iou  = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred_flat.sum() + label_flat.sum() + 1e-6)
    return iou.item(), dice.item()


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME, do_reduce_labels=False)

    train_ds = HouseSegDataset("train", processor)
    val_ds   = HouseSegDataset("val",   processor)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={0: "background", 1: "house"},
        label2id={"background": 0, "house": 1},
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = []
    best_iou = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for pixel_values, labels in train_dl:
            pixel_values = pixel_values.to(DEVICE)
            labels       = labels.to(DEVICE)
            outputs      = model(pixel_values=pixel_values, labels=labels)
            loss         = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss  += loss.item()
        train_loss /= len(train_dl)
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss  = 0.0
        iou_sum   = 0.0
        dice_sum  = 0.0
        with torch.no_grad():
            for pixel_values, labels in val_dl:
                pixel_values = pixel_values.to(DEVICE)
                labels       = labels.to(DEVICE)
                outputs      = model(pixel_values=pixel_values, labels=labels)
                val_loss    += outputs.loss.item()

                logits    = outputs.logits                          # (B, C, H/4, W/4)
                upsampled = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                preds     = upsampled.argmax(dim=1)
                iou, dice = compute_metrics(preds, labels)
                iou_sum  += iou
                dice_sum += dice

        val_loss /= len(val_dl)
        mean_iou  = iou_sum  / len(val_dl)
        mean_dice = dice_sum / len(val_dl)

        row = dict(epoch=epoch, train_loss=round(train_loss,4),
                   val_loss=round(val_loss,4),
                   iou=round(mean_iou,4), dice=round(mean_dice,4))
        history.append(row)
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"IoU={mean_iou:.4f}  Dice={mean_dice:.4f}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            model.save_pretrained(CHECKPOINT_DIR)
            processor.save_pretrained(CHECKPOINT_DIR)
            print(f"   >> New best IoU={best_iou:.4f} -- checkpoint saved")

    with open(CHECKPOINT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")
    print(f"Checkpoint: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    train()
