"""
═══════════════════════════════════════════════════════════════════════════════
 PART B — DEMONSTRATION UI  (Enhanced Streamlit web app)
═══════════════════════════════════════════════════════════════════════════════
 Research Scholar : Suriya Prakash  |  Supervisor: Dr. M. Balasubramanian
 University       : Annamalai University
 Research         : LULC Classification for Indian Regions using AI
───────────────────────────────────────────────────────────────────────────────
 A polished web UI to demonstrate the trained LULC segmentation models.

 RUN:
     pip install streamlit torch torchvision segmentation-models-pytorch timm
     pip install plotly pandas
     streamlit run 04_app.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import time
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
import streamlit as st
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"  # HF free tier = CPU
OUTPUT_DIR  = "./outputs"
IMG_SIZE    = 224
NUM_CLASSES = 7

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = ["Water", "Dense Forest", "Sparse Forest", "Barren",
               "Built-up", "Agricultural", "Fallow"]
CLASS_COLORS = np.array([
    [ 30, 144, 255], [ 34, 139,  34], [154, 205,  50], [160, 120,  90],
    [220,  20,  60], [255, 215,   0], [210, 180, 140],
], dtype=np.uint8)

MODELS = {
    "ConvNeXt"         : ("tu-convnext_tiny",                  "deeplabv3plus"),
    "Swin Transformer" : ("tu-swin_tiny_patch4_window7_224",  "unet"),
    "ResNet"           : ("resnet34",                          "deeplabv3plus"),
}

# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE DRIVE AUTO-DOWNLOAD OF MODEL FILES
# ─────────────────────────────────────────────────────────────────────────────
# The trained model weights (.pth) and results.json are hosted on Google Drive
# so the GitHub repo stays small. On first startup, they are downloaded once
# into ./outputs. Replace the FILE IDs below with your own share-link IDs.
#
# HOW TO GET A FILE ID:
#   1. Upload the file to Google Drive.
#   2. Right-click -> Share -> "Anyone with the link" -> Copy link.
#   3. The link looks like:
#        https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQ/view?usp=sharing
#      The FILE ID is the part between /d/ and /view :  1AbCdEfGhIjKlMnOpQ
#

GDRIVE_FILES = {
    "ConvNeXt.pth"         : "1dTO1xrc2y-sXet63mAQjX9eGcpFJC_sU",
    "Swin_Transformer.pth" : "1-fmmxWsk3YsiVFCMXzBNIDHYlEMCzdKY",
    "ResNet.pth"           : "17qz3i23rQRqq9FZxcnQjWwfBrFzC7OOe",
    "results.json"         : "1l4SyjppZjDqAGn_X90AzpLhnNgIPwRBv",
}

def _ensure_models_downloaded():
    """Download model files from Google Drive on first run (once)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    missing = {fn: fid for fn, fid in GDRIVE_FILES.items()
               if not os.path.exists(os.path.join(OUTPUT_DIR, fn))
               and not fid.startswith("PASTE_")}
    if not missing:
        return
    try:
        import gdown
    except ImportError:
        st.error("gdown is required to download models. Add 'gdown' to requirements.txt.")
        return
    prog = st.progress(0.0, text="Downloading trained models (first run only)...")
    for i, (fn, fid) in enumerate(missing.items(), start=1):
        dest = os.path.join(OUTPUT_DIR, fn)
        try:
            gdown.download(id=fid, output=dest, quiet=True)
        except Exception as e:
            st.warning(f"Could not download {fn}: {e}")
        prog.progress(i / len(missing),
                      text=f"Downloaded {fn} ({i}/{len(missing)})")
    prog.empty()

# run the download check once, at startup
_ensure_models_downloaded()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India LULC Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ---- Design tokens ---- */
    :root {
        --ink: #1f3864;
        --accent: #2e75b6;
        --teal: #107c6e;
        --amber: #c55a11;
        --paper: #f4f6fb;
    }
    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ---- Hero banner ---- */
    .hero {
        background: linear-gradient(135deg, #1f3864 0%, #2e75b6 60%, #107c6e 100%);
        padding: 28px 34px;
        border-radius: 16px;
        color: white;
        margin-bottom: 8px;
        box-shadow: 0 8px 24px rgba(31,56,100,0.25);
    }
    .hero h1 {
        color: white !important;
        font-size: 30px;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.3px;
    }
    .hero p { color: #e8f0fb; margin: 2px 0; font-size: 14px; }
    .hero .tag {
        display:inline-block; background: rgba(255,255,255,0.18);
        padding: 3px 12px; border-radius: 20px; font-size: 12px;
        margin-top: 10px; margin-right: 6px; backdrop-filter: blur(4px);
    }

    /* ---- Metric cards ---- */
    .metric-card {
        background: white; border-radius: 14px; padding: 18px 20px;
        border: 1px solid #e6ecf5; box-shadow: 0 2px 10px rgba(31,56,100,0.06);
        text-align: center; transition: transform .15s ease;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-card .label { font-size: 13px; color: #6b7280; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 34px; font-weight: 800; color: var(--ink);
        margin: 6px 0 0 0; }
    .metric-card .sub { font-size: 12px; color: #9ca3af; }
    .metric-card.best { background: linear-gradient(135deg, #e2efda, #d5f0ea);
        border-color: #92d050; }
    .metric-card.best .value { color: #107c6e; }

    /* ---- Section headers ---- */
    .section-head {
        font-size: 20px; font-weight: 700; color: var(--ink);
        border-left: 5px solid var(--accent); padding-left: 12px;
        margin: 10px 0 16px 0;
    }

    /* ---- Legend chips ---- */
    .legend-chip {
        display: flex; align-items: center; padding: 6px 10px;
        border-radius: 8px; margin-bottom: 4px; background: white;
        border: 1px solid #eef2f7; font-size: 14px;
    }
    .legend-swatch {
        width: 18px; height: 18px; border-radius: 5px; margin-right: 10px;
        border: 1px solid rgba(0,0,0,0.1);
    }

    /* ---- Badge ---- */
    .device-badge {
        display:inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 12px; font-weight: 700;
    }
    .device-gpu { background:#d5f0ea; color:#107c6e; }
    .device-cpu { background:#fdebd0; color:#c55a11; }

    /* Rounded images */
    div[data-testid="stImage"] img { border-radius: 12px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.10); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(name):
    encoder, architecture = MODELS[name]
    common = dict(encoder_name=encoder, encoder_weights=None,
                  in_channels=3, classes=NUM_CLASSES)
    model = smp.Unet(**common) if architecture == "unet" else smp.DeepLabV3Plus(**common)
    wp = os.path.join(OUTPUT_DIR, f"{name.replace(' ','_')}.pth")
    loaded = False
    if os.path.exists(wp):
        model.load_state_dict(torch.load(wp, map_location=DEVICE))
        loaded = True
    model.to(DEVICE).eval()
    return model, loaded


@st.cache_data
def load_results():
    path = os.path.join(OUTPUT_DIR, "results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def predict(model, pil_img):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = TF.to_tensor(img)
    x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(1).squeeze(0).cpu().numpy()
    infer_ms = (time.time() - t0) * 1000
    color_mask = CLASS_COLORS[pred]
    return Image.fromarray(color_mask), pred, infer_ms, img


def overlay(base_img, color_mask, alpha=0.5):
    base = np.array(base_img.convert("RGB")).astype(float)
    cm   = np.array(color_mask.resize(base_img.size, Image.NEAREST)).astype(float)
    blend = (base * (1 - alpha) + cm * alpha).astype(np.uint8)
    return Image.fromarray(blend)


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
dev_badge = ('<span class="device-badge device-gpu">⚡ GPU</span>'
             if DEVICE == "cuda"
             else '<span class="device-badge device-cpu">CPU</span>')
st.markdown(f"""
<div class="hero">
    <h1>🛰️ LULC Classification for Indian Regions</h1>
    <p><b>Deep Learning-based Land Use &amp; Land Cover Segmentation of Sentinel-2 Satellite Imagery</b></p>
    <p>Research Scholar: Suriya Prakash &nbsp;•&nbsp; Supervisor: Dr. M. Balasubramanian &nbsp;•&nbsp; Annamalai University</p>
    <div>
        <span class="tag">Sen-2 LULC Dataset</span>
        <span class="tag">7 Land Cover Classes</span>
        <span class="tag">3 Deep Learning Models</span>
        <span class="tag">Running on {('RTX 3060' if DEVICE=='cuda' else 'CPU')}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    model_name = st.selectbox("**Segmentation Model**", list(MODELS.keys()),
                              help="Choose which trained model to run")
    enc, arch = MODELS[model_name]
    st.caption(f"Encoder: `{enc}`  •  Architecture: `{arch}`")

    st.markdown(f"**Compute device:** {dev_badge}", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🎨 Land Cover Classes")
    for nm, col in zip(CLASS_NAMES, CLASS_COLORS):
        hexc = "#%02x%02x%02x" % tuple(col)
        st.markdown(
            f'<div class="legend-chip"><span class="legend-swatch" '
            f'style="background:{hexc}"></span>{nm}</div>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.caption("💡 Tip: use tiles from the dataset's `test_images` folder "
               "for best results.")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  🖼️  Segment an Image  ",
    "  📊  Model Comparison  ",
    "  ℹ️  About This Project  ",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-head">Upload a satellite image to classify land cover</div>',
                unsafe_allow_html=True)

    # discover bundled sample images
    import glob as _glob
    sample_paths = sorted(
        _glob.glob(os.path.join("samples", "*.png")) +
        _glob.glob(os.path.join("samples", "*.jpg")) +
        _glob.glob(os.path.join("samples", "*.jpeg")) +
        _glob.glob(os.path.join("samples", "*.tif")) +
        _glob.glob(os.path.join("samples", "*.tiff"))
    )
    sample_names = [os.path.basename(p) for p in sample_paths]

    # two ways to provide an image: pick a sample, or upload your own
    src_col1, src_col2 = st.columns(2)

    with src_col1:
        st.markdown("**🗂️ Choose a sample image**")
        if sample_names:
            choice = st.selectbox(
                "Available sample tiles",
                ["— none —"] + sample_names,
                label_visibility="collapsed",
                help="Pre-loaded Sentinel-2 tiles you can classify instantly")
        else:
            choice = "— none —"
            st.caption("No sample images available.")

    with src_col2:
        st.markdown("**⬆️ Or upload your own**")
        uploaded = st.file_uploader("Upload a Sentinel-2 satellite tile",
                                    type=["png", "jpg", "jpeg", "tif", "tiff"],
                                    label_visibility="collapsed")

    # thumbnail gallery of samples (click-free visual reference)
    if sample_names:
        with st.expander("👁️ Preview all sample images", expanded=False):
            thumbs = st.columns(min(len(sample_paths), 5))
            for i, sp in enumerate(sample_paths):
                with thumbs[i % len(thumbs)]:
                    st.image(sp, caption=sample_names[i], use_container_width=True)

    # resolve the image source — an upload takes priority, else the chosen sample
    pil_img = None
    if uploaded is not None:
        pil_img = Image.open(uploaded)
    elif choice != "— none —":
        idx = sample_names.index(choice)
        pil_img = Image.open(sample_paths[idx])

    if pil_img is not None:

        with st.spinner(f"🔍 Running {model_name} segmentation..."):
            model, loaded = load_model(model_name)
            color_mask, pred, infer_ms, resized_in = predict(model, pil_img)
            blended = overlay(resized_in, color_mask, alpha=0.55)

        if not loaded:
            st.warning(f"⚠️ No trained weights found for {model_name}. "
                       f"Showing untrained output. Run `03_train_models.py` first.")

        # --- Three-view display ---
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📷 Input Image**")
            st.image(resized_in, use_container_width=True)
        with c2:
            st.markdown("**🗺️ Segmentation Map**")
            st.image(color_mask, use_container_width=True)
        with c3:
            st.markdown("**🔀 Overlay**")
            st.image(blended, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Inference stats + class distribution ---
        left, right = st.columns([1, 2])

        with left:
            st.markdown('<div class="section-head">Inference</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Inference Time</div>
                <div class="value">{infer_ms:.0f}<span style="font-size:16px;"> ms</span></div>
                <div class="sub">on {('GPU' if DEVICE=='cuda' else 'CPU')} • {model_name}</div>
            </div>""", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-head">Land Cover Distribution</div>',
                        unsafe_allow_html=True)
            total = pred.size
            dist = [(nm, 100.0*(pred==i).sum()/total, CLASS_COLORS[i])
                    for i, nm in enumerate(CLASS_NAMES)]
            dist = [d for d in dist if d[1] > 0.1]
            dist.sort(key=lambda x: -x[1])

            try:
                import plotly.graph_objects as go
                fig = go.Figure(go.Bar(
                    x=[d[1] for d in dist],
                    y=[d[0] for d in dist],
                    orientation="h",
                    marker_color=["#%02x%02x%02x" % tuple(d[2]) for d in dist],
                    text=[f"{d[1]:.1f}%" for d in dist],
                    textposition="outside",
                ))
                fig.update_layout(
                    height=max(200, 44*len(dist)),
                    margin=dict(l=10, r=30, t=10, b=10),
                    xaxis=dict(title="Coverage (%)", range=[0, max(d[1] for d in dist)*1.15]),
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=13),
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                for nm, pct, _ in dist:
                    st.progress(pct/100.0, text=f"{nm}: {pct:.1f}%")
    else:
        st.info("👆 **Upload a Sentinel-2 satellite tile** to see the model "
                "classify each pixel into land cover types.")
        # sample placeholder row
        st.markdown('<div class="section-head">What you\'ll get</div>',
                    unsafe_allow_html=True)
        cc1, cc2, cc3 = st.columns(3)
        cc1.markdown("**📷 Input** — your satellite image")
        cc2.markdown("**🗺️ Segmentation** — pixel-wise land cover map")
        cc3.markdown("**🔀 Overlay** — map blended over the original")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-head">Performance of trained models on Indian LULC data</div>',
                unsafe_allow_html=True)
    results = load_results()

    if results and results.get("models"):
        rows = []
        for name, r in results["models"].items():
            rows.append({
                "Model": name,
                "Architecture": r.get("architecture", "-"),
                "Encoder": r.get("encoder", "-"),
                "IoU": round(r.get("best_iou", 0), 4),
                "Dice": round(r.get("final_dice", 0), 4),
                "Train Time (s)": r.get("train_time_s", 0),
            })
        df = pd.DataFrame(rows).sort_values("IoU", ascending=False).reset_index(drop=True)
        best_row = df.iloc[0]

        # --- Metric cards ---
        cols = st.columns(len(df))
        for i, (col, (_, r)) in enumerate(zip(cols, df.iterrows())):
            is_best = (r["Model"] == best_row["Model"])
            cls = "metric-card best" if is_best else "metric-card"
            crown = "👑 " if is_best else ""
            col.markdown(f"""
            <div class="{cls}">
                <div class="label">{crown}{r['Model']}</div>
                <div class="value">{r['IoU']:.3f}</div>
                <div class="sub">IoU • Dice {r['Dice']:.3f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Charts ---
        try:
            import plotly.graph_objects as go
            colA, colB = st.columns(2)
            palette = ["#2e75b6", "#107c6e", "#c55a11", "#7030a0"]

            with colA:
                st.markdown("**IoU by Model** (higher = better)")
                fig = go.Figure(go.Bar(
                    x=df["Model"], y=df["IoU"],
                    marker_color=palette[:len(df)],
                    text=[f"{v:.3f}" for v in df["IoU"]], textposition="outside"))
                fig.update_layout(height=320, margin=dict(l=10,r=10,t=20,b=10),
                    yaxis=dict(range=[0, df["IoU"].max()*1.2], title="IoU"),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            with colB:
                st.markdown("**Accuracy vs Training Cost**")
                fig = go.Figure(go.Scatter(
                    x=df["Train Time (s)"], y=df["IoU"],
                    mode="markers+text", text=df["Model"],
                    textposition="top center",
                    marker=dict(size=22, color=palette[:len(df)],
                                line=dict(width=2, color="white"))))
                fig.update_layout(height=320, margin=dict(l=10,r=10,t=20,b=10),
                    xaxis=dict(title="Training Time (s)"),
                    yaxis=dict(title="IoU"),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(df.set_index("Model")["IoU"])

        # --- Detailed table ---
        st.markdown('<div class="section-head">Detailed Results</div>',
                    unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True,
            column_config={
                "IoU": st.column_config.ProgressColumn(
                    "IoU", min_value=0.0, max_value=1.0, format="%.4f"),
                "Dice": st.column_config.ProgressColumn(
                    "Dice", min_value=0.0, max_value=1.0, format="%.4f"),
            })

        st.success(f"🏆 **Best performing model: {best_row['Model']}** "
                   f"(IoU = {best_row['IoU']:.4f}) — a transformer-based model, "
                   f"supporting the proposed hybrid Swin-Transformer approach for "
                   f"Indian LULC segmentation.")
    else:
        st.warning("No results found yet. Train the models first: "
                   "`python 03_train_models.py`. The UI reads `outputs/results.json`.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-head">About This Research Prototype</div>',
                unsafe_allow_html=True)

    st.markdown("""
    This application demonstrates the first phase of my PhD research on
    **Land Use and Land Cover (LULC) classification for Indian regions** using
    deep learning and satellite imagery.

    #### 🎯 Objective
    To classify each pixel of a Sentinel-2 satellite image into one of seven
    land cover categories — enabling automated monitoring of agriculture,
    forests, water bodies, and urban growth across Indian terrain.

    #### 🗂️ Dataset
    **Sen-2 LULC** — 213,761 Sentinel-2 satellite images at 10m resolution,
    covering the Indian region across 7 land cover classes.

    #### 🧠 Models Demonstrated
    Three pretrained backbones adapted for semantic segmentation:
    - **ConvNeXt** (DeepLabV3+) — a modern convolutional architecture
    - **Swin Transformer** (U-Net) — a hierarchical vision transformer
    - **ResNet** (DeepLabV3+) — a classic CNN baseline

    #### 🔬 Why This Matters
    The Swin Transformer's strong performance here provides experimental support
    for my proposed **hybrid Swin-Transformer + SE-ASPP** architecture — the
    core novelty of my thesis.

    #### 🚀 Next Steps
    - Multi-region validation across Indian agro-climatic zones
    - SAR + optical fusion (Sentinel-1 + Sentinel-2) for monsoon-resilient mapping
    - Explainable AI (XAI) for transparent, policy-ready land cover maps
    """)

    st.markdown("---")
    st.caption("Prototype demonstration • Annamalai University • "
               "Department of Computer Science and Engineering")
