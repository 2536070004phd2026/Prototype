# LULC Classification for Indian Regions — Prototype (Parts A & B)

**Research Scholar:** Suriya Prakash
**Supervisor:** Dr. M. Balasubramanian
**University:** Annamalai University
**Hardware:** NVIDIA RTX 3060 Laptop (6 GB VRAM)

A one-week prototype: train deep learning models on the Sen-2 LULC Indian
dataset (Part A) and demonstrate them in a web UI (Part B).

---

## One-Time Setup

Install everything (in a virtual environment):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install segmentation-models-pytorch timm streamlit
pip install scikit-learn numpy pillow pandas tqdm
```

> The `cu121` index gets the CUDA build so your RTX 3060 is used. Verify with:
> ```python
> import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
> ```

---

## The Files

| File | Purpose |
|------|---------|
| `01_download_dataset.py` | Instructions + check that the dataset is in place |
| `02_data_pipeline.py`    | Dataset class + train/val/test DataLoaders |
| `03_train_models.py`     | Train 3 segmentation models (GPU, mixed precision) |
| `04_app.py`              | Streamlit demonstration UI |

---

## 7-Day Plan

### Days 1–2 — Data (Part A)
1. Download Sen-2 LULC from https://data.mendeley.com/datasets/fdk57tc57m
   and extract into a folder named `sen2lulc`.
2. Run `python 01_download_dataset.py` to verify it's found.
3. Open `02_data_pipeline.py`, check `IMAGE_DIR` / `MASK_DIR` match the real
   folder layout, then run `python 02_data_pipeline.py` — it should print one
   batch's shapes. That confirms data loads correctly.

### Days 3–5 — Train (Part A)
4. Run `python 03_train_models.py`.
   - Trains ConvNeXt, Swin Transformer, ResNet on the subset.
   - Uses batch size 8 + mixed precision (safe for 6 GB).
   - Saves weights + `outputs/results.json`.
   - If you see "out of memory": lower `BATCH_SIZE` to 4 in
     `02_data_pipeline.py`.

### Days 6–7 — UI (Part B)
5. Run `streamlit run 04_app.py`.
   - Tab 1: upload a satellite tile → see the colour-coded LULC map.
   - Tab 2: compare model IoU / Dice / training time.

---

## VRAM Tips (6 GB)

- Keep `BATCH_SIZE = 8` (or 4 if needed).
- Mixed precision is already on (halves memory).
- Using "tiny" encoders (convnext_tiny, swin_tiny) — right-sized for 6 GB.
- Close other GPU apps (games, browsers with GPU acceleration) while training.

---

## What to Show Your Guide

> "A working prototype: 3 deep learning models trained on the Sen-2 LULC
> Indian dataset for pixel-level land cover segmentation, with a web UI that
> runs any model on a satellite image and compares their performance.
> Next steps: multi-region validation, SAR fusion, and XAI integration."

That is a strong, honest one-week deliverable.
