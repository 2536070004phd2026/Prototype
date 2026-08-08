"""
═══════════════════════════════════════════════════════════════════════════════
 PART A — STEP 2: DATA PIPELINE  (matches confirmed Sen-2 LULC folder layout)
═══════════════════════════════════════════════════════════════════════════════
 Research Scholar : Suriya Prakash  |  Supervisor: Dr. M. Balasubramanian
 University       : Annamalai University
 Hardware         : NVIDIA RTX 3060 Laptop (6 GB VRAM)
───────────────────────────────────────────────────────────────────────────────
 The Sen-2 LULC dataset is ALREADY split into six folders:

     sen2lulc/SEN-2 LULC/
         ├── train_images/   ├── train_masks/
         ├── test_images/     ├── test_masks/
         └── val_images/      └── val_masks/

 So we use those folders directly (no manual splitting needed).
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# Point to the inner "SEN-2 LULC" folder (note the space in the name).
DATA_ROOT = os.path.join("sen2lulc", "SEN-2 LULC")

# NOTE: each folder has an inner subfolder (train_images/train, etc.)
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images", "train")
TRAIN_MSK_DIR = os.path.join(DATA_ROOT, "train_masks",  "train")
VAL_IMG_DIR   = os.path.join(DATA_ROOT, "val_images",   "val")
VAL_MSK_DIR   = os.path.join(DATA_ROOT, "val_masks",    "val")
TEST_IMG_DIR  = os.path.join(DATA_ROOT, "test_images",  "test")
TEST_MSK_DIR  = os.path.join(DATA_ROOT, "test_masks",   "test")

NUM_CLASSES = 7
IMG_SIZE    = 224
BATCH_SIZE  = 8            # lower to 4 if out-of-memory on 6 GB
SUBSET_SIZE = 3000         # cap train images for prototype; None = all
SEED        = 42
NUM_WORKERS = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = ["Water", "Dense Forest", "Sparse Forest", "Barren",
               "Built-up", "Agricultural", "Fallow"]
CLASS_COLORS = np.array([
    [ 30, 144, 255],
    [ 34, 139,  34],
    [154, 205,  50],
    [160, 120,  90],
    [220,  20,  60],
    [255, 215,   0],
    [210, 180, 140],
], dtype=np.uint8)


class Sen2LULCDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=IMG_SIZE, train=True, limit=None):
        self.img_size = img_size
        self.train    = train

        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
        imgs = []
        for e in exts:
            imgs += glob.glob(os.path.join(img_dir, e))
        imgs = sorted(imgs)

        self.pairs = []
        for ip in imgs:
            fname = os.path.basename(ip)
            mp = os.path.join(mask_dir, fname)
            if os.path.exists(mp):
                self.pairs.append((ip, mp))
            else:
                stem = os.path.splitext(fname)[0]
                cand = []
                for e in exts:
                    cand += glob.glob(os.path.join(mask_dir, stem + e.replace("*", "")))
                if cand:
                    self.pairs.append((ip, cand[0]))

        if limit is not None:
            self.pairs = self.pairs[:limit]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]

        img = Image.open(ip).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.open(mp).resize(
            (self.img_size, self.img_size), Image.NEAREST)
        mask = np.array(mask)

        if mask.ndim == 3:
            mask = self._rgb_to_class(mask)
        mask = mask.astype(np.int64)
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        img = TF.to_tensor(img)
        if self.train and torch.rand(1).item() < 0.5:
            img  = TF.hflip(img)
            mask = np.fliplr(mask).copy()

        img  = TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
        mask = torch.from_numpy(mask).long()
        return img, mask

    @staticmethod
    def _rgb_to_class(rgb_mask):
        h, w, _ = rgb_mask.shape
        out = np.zeros((h, w), dtype=np.int64)
        for cls_idx, color in enumerate(CLASS_COLORS):
            matches = np.all(rgb_mask[:, :, :3] == color, axis=-1)
            out[matches] = cls_idx
        return out


def get_dataloaders():
    torch.manual_seed(SEED)

    for d in [TRAIN_IMG_DIR, TRAIN_MSK_DIR, VAL_IMG_DIR, VAL_MSK_DIR,
              TEST_IMG_DIR, TEST_MSK_DIR]:
        if not os.path.isdir(d):
            raise RuntimeError(
                f"Folder not found: {d}\n"
                f"  -> Check DATA_ROOT at the top of this file. It should point "
                f"to the inner 'SEN-2 LULC' folder that contains the six "
                f"train/val/test folders."
            )

    train_ds = Sen2LULCDataset(TRAIN_IMG_DIR, TRAIN_MSK_DIR, train=True,  limit=SUBSET_SIZE)
    val_lim  = None if SUBSET_SIZE is None else max(1, SUBSET_SIZE // 6)
    val_ds   = Sen2LULCDataset(VAL_IMG_DIR,   VAL_MSK_DIR,   train=False, limit=val_lim)
    test_ds  = Sen2LULCDataset(TEST_IMG_DIR,  TEST_MSK_DIR,  train=False, limit=val_lim)

    if len(train_ds) == 0:
        raise RuntimeError(
            f"Found 0 image/mask pairs in {TRAIN_IMG_DIR}.\n"
            f"  -> Open that folder and check the image file names match the "
            f"mask file names in {TRAIN_MSK_DIR}."
        )

    print(f"Pairs -> train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Building data loaders (quick self-test)...")
    try:
        tr, va, te = get_dataloaders()
        imgs, masks = next(iter(tr))
        print(f"\nOne training batch:")
        print(f"  images: {imgs.shape}  (B, C, H, W)")
        print(f"  masks : {masks.shape}  (B, H, W)")
        print(f"  mask class values present: {torch.unique(masks).tolist()}")
        print("\n[OK] Data pipeline is working. Next: run 03_train_models.py")
    except Exception as e:
        print(f"\n[!] {e}")