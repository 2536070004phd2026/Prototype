"""
═══════════════════════════════════════════════════════════════════════════════
 PART A — STEP 3: TRAIN SEGMENTATION MODELS  (RTX 3060, 6 GB VRAM)
═══════════════════════════════════════════════════════════════════════════════
 Research Scholar : Suriya Prakash  |  Supervisor: Dr. M. Balasubramanian
 University       : Annamalai University
 Research         : LULC Classification for Indian Regions using AI
───────────────────────────────────────────────────────────────────────────────
 Trains 3 pretrained backbones adapted for LULC SEMANTIC SEGMENTATION using
 the DeepLabV3+ framework, on the Sen-2 LULC Indian dataset.

 Models (chosen from the earlier 9-model study as the strongest + a baseline):
     1. ConvNeXt   (top performer in the classification study)
     2. Swin Transformer (best accuracy-to-size; core to the proposed model)
     3. ResNet     (classic baseline; used in the base paper)

 VRAM-SAFE SETTINGS for 6 GB:
     - batch size 8
     - mixed precision (AMP) -> ~half the memory
     - "tiny" backbone variants

 INSTALL (one time):
     pip install torch torchvision segmentation-models-pytorch timm
     pip install scikit-learn numpy pillow tqdm

 Trained weights + a results.json are saved to ./outputs for the UI (Part B).
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import segmentation_models_pytorch as smp

from importlib import import_module
dp = import_module("02_data_pipeline")   # reuse our data pipeline
get_dataloaders = dp.get_dataloaders
NUM_CLASSES = dp.NUM_CLASSES
CLASS_NAMES = dp.CLASS_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS      = 15           # good for a prototype on GPU; raise later
LR          = 1e-4
OUTPUT_DIR  = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Each model: (encoder_name, architecture).
# DeepLabV3+ works for CNN encoders (ConvNeXt, ResNet).
# Swin is a hierarchical transformer whose feature strides do NOT fit
# DeepLabV3+'s decoder, so we use U-Net for it (U-Net handles Swin correctly).
MODELS = {
    "ConvNeXt"         : ("tu-convnext_tiny",                  "deeplabv3plus"),
    "Swin Transformer" : ("tu-swin_tiny_patch4_window7_224",  "unet"),
    "ResNet"           : ("resnet34",                          "deeplabv3plus"),
}


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_iou(preds, masks, num_classes=NUM_CLASSES):
    """Mean IoU across classes for a batch."""
    ious = []
    preds = preds.view(-1)
    masks = masks.view(-1)
    for c in range(num_classes):
        pred_c = (preds == c)
        mask_c = (masks == c)
        inter  = (pred_c & mask_c).sum().item()
        union  = (pred_c | mask_c).sum().item()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def dice_score(preds, masks, num_classes=NUM_CLASSES):
    dices = []
    preds = preds.view(-1)
    masks = masks.view(-1)
    for c in range(num_classes):
        pred_c = (preds == c)
        mask_c = (masks == c)
        inter  = (pred_c & mask_c).sum().item()
        denom  = pred_c.sum().item() + mask_c.sum().item()
        if denom > 0:
            dices.append(2 * inter / denom)
    return float(np.mean(dices)) if dices else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────────────────────────────────────
def build_model(encoder_name, architecture="deeplabv3plus"):
    """Build a segmentation model with a pretrained encoder for NUM_CLASSES.
    architecture: 'deeplabv3plus' (CNN encoders) or 'unet' (works with Swin)."""
    common = dict(encoder_name=encoder_name, encoder_weights="imagenet",
                  in_channels=3, classes=NUM_CLASSES)
    if architecture == "unet":
        model = smp.Unet(**common)
    else:
        model = smp.DeepLabV3Plus(**common)
    return model.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train_one(name, encoder_name, architecture, train_loader, val_loader):
    print(f"\n{'='*70}\n  Training {name}  (encoder: {encoder_name}, arch: {architecture})\n{'='*70}")
    model = build_model(encoder_name, architecture)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler    = GradScaler("cuda")   # mixed precision for 6GB VRAM

    best_iou = 0.0
    t0 = time.time()

    for epoch in range(EPOCHS):
        # ---- train ----
        model.train()
        running = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            with autocast("cuda"):                # mixed precision
                out  = model(imgs)
                loss = criterion(out, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        # ---- validate ----
        model.eval()
        v_iou, v_dice, n = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                with autocast("cuda"):
                    out = model(imgs)
                preds = out.argmax(1)
                v_iou  += compute_iou(preds, masks)
                v_dice += dice_score(preds, masks)
                n += 1
        v_iou  /= max(n, 1)
        v_dice /= max(n, 1)
        print(f"  Epoch {epoch+1:2d}/{EPOCHS}  loss={running/len(train_loader):.4f}"
              f"  val_IoU={v_iou:.4f}  val_Dice={v_dice:.4f}")

        # save best
        if v_iou > best_iou:
            best_iou = v_iou
            torch.save(model.state_dict(),
                       os.path.join(OUTPUT_DIR, f"{name.replace(' ','_')}.pth"))

    elapsed = time.time() - t0
    print(f"  Done. best val IoU = {best_iou:.4f}  ({elapsed:.0f}s)")
    return {"encoder": encoder_name, "architecture": architecture,
            "best_iou": best_iou, "final_dice": v_dice,
            "train_time_s": round(elapsed)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    train_loader, val_loader, test_loader = get_dataloaders()

    results = {}
    for name, (encoder, architecture) in MODELS.items():
        try:
            results[name] = train_one(name, encoder, architecture,
                                      train_loader, val_loader)
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [!] OUT OF MEMORY for {name}. "
                      f"Lower BATCH_SIZE in 02_data_pipeline.py (try 4).")
                torch.cuda.empty_cache()
            else:
                raise

    # save all results + class names for the UI
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump({"models": results, "class_names": CLASS_NAMES},
                  f, indent=2)

    print(f"\n{'='*70}")
    print("  ALL DONE. Results saved to outputs/results.json")
    print("  Trained weights saved as outputs/<ModelName>.pth")
    print("  Next: run the Streamlit UI (Part B) -> streamlit run 04_app.py")
    print(f"{'='*70}")
    for name, r in results.items():
        print(f"  {name:<20} IoU={r['best_iou']:.4f}  time={r['train_time_s']}s")


if __name__ == "__main__":
    main()