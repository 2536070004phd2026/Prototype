"""
═══════════════════════════════════════════════════════════════════════════════
 PART A — STEP 1: DATASET DOWNLOAD & SETUP
═══════════════════════════════════════════════════════════════════════════════
 Research Scholar : Suriya Prakash
 Supervisor       : Dr. M. Balasubramanian
 University       : Annamalai University
 Research         : LULC Classification for Indian Regions using AI
 Hardware         : NVIDIA RTX 3060 Laptop (6 GB VRAM)
───────────────────────────────────────────────────────────────────────────────

 This script prepares the Sen-2 LULC Indian dataset for training.

 The Sen-2 LULC dataset (Sawant et al., 2023) contains 213,761 Sentinel-2
 satellite images at 10m resolution covering 7 LULC classes across the
 Indian region:
     0 = Water bodies
     1 = Dense forest
     2 = Sparse forest
     3 = Barren land
     4 = Built-up
     5 = Agricultural land
     6 = Fallow land

 ─── HOW TO DOWNLOAD (do this manually, one time) ───────────────────────────

 The dataset is hosted on Mendeley Data (free, open access):

   1. Go to:  https://data.mendeley.com/datasets/fdk57tc57m
      (Search "Sen-2 LULC dataset" on data.mendeley.com if the link changes)

   2. Click "Download All" — it downloads a ZIP (~a few GB).

   3. Extract the ZIP into a folder named 'sen2lulc' next to these scripts.
      After extraction you should have image tiles and their masks.

   4. Update DATA_ROOT below to point to that folder.

 NOTE: The dataset is large (213k images). For our 1-week prototype we will
 only use a SUBSET (controlled by SUBSET_SIZE in the training script), which
 is plenty to demonstrate the models and is friendly to 6 GB VRAM.

 ─── ALTERNATIVE: Kaggle mirror ─────────────────────────────────────────────
 The dataset is also mirrored on Kaggle. If you have a Kaggle account:
     pip install kaggle
     kaggle datasets download -d <sen2-lulc-dataset-slug>
 (Search "Sen-2 LULC" on kaggle.com/datasets for the current slug.)
═══════════════════════════════════════════════════════════════════════════════
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — update this to your actual dataset path
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT = "./sen2lulc"   # folder where you extracted the dataset

def verify_dataset():
    """Check that the dataset folder exists and report what's inside."""
    if not os.path.isdir(DATA_ROOT):
        print("=" * 70)
        print("  DATASET NOT FOUND")
        print("=" * 70)
        print(f"  Expected folder: {os.path.abspath(DATA_ROOT)}")
        print()
        print("  Please download the Sen-2 LULC dataset first:")
        print("  1. Visit: https://data.mendeley.com/datasets/fdk57tc57m")
        print("  2. Download All and extract into a folder named 'sen2lulc'")
        print("  3. Place that folder next to this script")
        print("  4. Re-run this script to verify")
        print("=" * 70)
        return False

    print("=" * 70)
    print("  DATASET FOLDER FOUND")
    print("=" * 70)
    print(f"  Location: {os.path.abspath(DATA_ROOT)}")
    print()
    print("  Top-level contents:")
    for item in sorted(os.listdir(DATA_ROOT))[:20]:
        full = os.path.join(DATA_ROOT, item)
        kind = "DIR " if os.path.isdir(full) else "FILE"
        print(f"    [{kind}] {item}")
    print()
    print("  Next step: run 02_data_pipeline.py to build the data loaders.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    verify_dataset()
