from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import ModelUnavailableError, load_artifacts, normalize_mask, run_inference


st.set_page_config(page_title="Building Footprint", layout="wide")


def apply_theme() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@500;600;700&family=Fira+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --app-bg: #020617;
    --app-bg-2: #050816;
    --panel: rgba(8, 13, 28, 0.88);
    --panel-strong: rgba(15, 23, 42, 0.92);
    --line: rgba(148, 163, 184, 0.18);
    --line-strong: rgba(125, 211, 252, 0.28);
    --text: #f8fafc;
    --muted: #94a3b8;
    --cyan: #38bdf8;
    --blue: #60a5fa;
    --violet: #8b5cf6;
    --green: #22c55e;
    --danger: #fb7185;
}

html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 52% 22%, rgba(96, 165, 250, 0.18), transparent 34rem),
        radial-gradient(circle at 74% 18%, rgba(34, 197, 94, 0.12), transparent 28rem),
        radial-gradient(circle at 32% 26%, rgba(139, 92, 246, 0.14), transparent 30rem),
        linear-gradient(180deg, #000000 0%, var(--app-bg) 42%, var(--app-bg-2) 100%);
    color: var(--text);
    font-family: 'Fira Sans', 'Segoe UI', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    display: none;
}

.block-container {
    max-width: 1480px;
    padding: 1.2rem 2.2rem 2.4rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(2, 6, 23, 0.98), rgba(8, 13, 28, 0.98));
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 1.2rem;
}

h1, h2, h3, p, label, span, div {
    font-family: 'Fira Sans', 'Segoe UI', sans-serif;
}

.topbar {
    min-height: 72px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    border-bottom: 1px solid var(--line);
}

.brand {
    display: flex;
    align-items: baseline;
    gap: 0.65rem;
}

.brand-name {
    color: var(--text);
    font-weight: 800;
    font-size: 1.18rem;
    letter-spacing: 0;
}

.brand-tag {
    color: var(--muted);
    font-family: 'Fira Code', Consolas, monospace;
    font-size: 0.78rem;
}

.nav {
    display: flex;
    align-items: center;
    gap: 1.45rem;
    color: #e5e7eb;
    font-weight: 700;
    font-size: 0.92rem;
}

.nav span {
    color: #e5e7eb;
}

.nav .active {
    color: var(--cyan);
}

.hero {
    min-height: 156px;
    display: grid;
    place-items: center;
    text-align: center;
    border-bottom: 1px solid var(--line);
}

.hero h1 {
    margin: 0;
    font-size: clamp(2rem, 4vw, 3.4rem);
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: 0;
    background: linear-gradient(90deg, var(--violet), var(--blue), #67e8f9);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero p {
    margin: 0.8rem 0 0;
    color: #e5e7eb;
    font-size: 1rem;
    font-weight: 600;
}

.status-row {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 1.1rem;
}

.status-pill {
    border: 1px solid var(--line-strong);
    background: rgba(15, 23, 42, 0.72);
    color: #dbeafe;
    border-radius: 999px;
    padding: 0.32rem 0.72rem;
    font-size: 0.76rem;
    font-family: 'Fira Code', Consolas, monospace;
}

.panel-title {
    margin: 0.4rem 0 0.75rem;
    color: var(--text);
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.section-title {
    margin: 1.1rem 0 0.6rem;
    color: var(--text);
    font-size: 0.92rem;
    font-weight: 800;
}

.empty-state {
    margin-top: 1.2rem;
    padding: 2.2rem;
    border: 1px dashed var(--line-strong);
    border-radius: 8px;
    background: rgba(15, 23, 42, 0.54);
    color: #dbeafe;
    text-align: center;
    font-weight: 700;
}

.run-caption {
    margin: 0.6rem 0 0.95rem;
    color: var(--muted);
    font-family: 'Fira Code', Consolas, monospace;
    font-size: 0.82rem;
}

.artifact-note {
    color: var(--muted);
    font-size: 0.82rem;
    line-height: 1.55;
    border-top: 1px solid var(--line);
    margin-top: 1rem;
    padding-top: 0.9rem;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.82), rgba(8, 13, 28, 0.82));
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.8rem 0.9rem;
    box-shadow: 0 18px 36px rgba(0, 0, 0, 0.22);
}

[data-testid="stMetric"] label {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: 'Fira Code', Consolas, monospace;
}

[data-testid="stImage"] {
    background: rgba(15, 23, 42, 0.56);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.65rem;
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.28);
}

[data-testid="stImage"] img {
    border-radius: 6px;
}

[data-testid="stFileUploader"] {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: rgba(15, 23, 42, 0.54);
    padding: 0.15rem 0.35rem;
}

[data-testid="stFileUploader"] section {
    border-color: rgba(125, 211, 252, 0.24) !important;
    background: rgba(2, 6, 23, 0.58) !important;
}

.stButton > button {
    min-height: 44px;
    border-radius: 8px;
    border: 1px solid rgba(56, 189, 248, 0.35);
    background: linear-gradient(90deg, var(--blue), var(--cyan));
    color: #020617;
    font-weight: 800;
    letter-spacing: 0;
    transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    filter: brightness(1.04);
    box-shadow: 0 14px 34px rgba(56, 189, 248, 0.24);
}

[data-baseweb="select"] > div,
[data-baseweb="slider"] {
    color: var(--text);
}

@media (max-width: 760px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }
    .topbar {
        align-items: flex-start;
        flex-direction: column;
        padding-bottom: 0.85rem;
    }
    .nav {
        gap: 0.9rem;
        flex-wrap: wrap;
        font-size: 0.82rem;
    }
    .hero {
        min-height: 132px;
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading model artifacts...")
def get_artifacts():
    return load_artifacts()


def read_rgb_upload(uploaded_file) -> np.ndarray:
    with Image.open(uploaded_file) as image:
        return np.asarray(image.convert("RGB"))


def read_mask_upload(uploaded_file, target_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(uploaded_file) as image:
        return normalize_mask(np.asarray(image.convert("L")), target_shape)


def metric_value(metrics: dict[str, float], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "-"
    return f"{value:.3f}"


def render_header(artifacts_device: str, threshold: float) -> None:
    st.markdown(
        f"""
<div class="topbar">
  <div class="brand">
    <span class="brand-name">Building Footprint</span>
    <span class="brand-tag">Project 16</span>
  </div>
  <div class="nav">
    <span class="active">Demo</span>
    <span>Methods</span>
    <span>Metrics</span>
    <span>Artifacts</span>
  </div>
</div>
<section class="hero">
  <div>
    <h1>Building Footprint Extraction</h1>
    <p>Aerial image segmentation demo</p>
    <div class="status-row">
      <span class="status-pill">U-Net device: {artifacts_device}</span>
      <span class="status-pill">threshold: {threshold:.2f}</span>
      <span class="status-pill">patch: 512 x 512</span>
    </div>
  </div>
</section>
        """,
        unsafe_allow_html=True,
    )


apply_theme()

artifacts = get_artifacts()
available_methods = artifacts.available_methods()
default_index = available_methods.index("U-Net") if "U-Net" in available_methods else 0

render_header(artifacts.device_label(), artifacts.unet_threshold)

with st.sidebar:
    st.markdown('<div class="panel-title">Controls</div>', unsafe_allow_html=True)
    method = st.selectbox("Method", available_methods, index=default_index)

    overlay_alpha = st.slider(
        "Overlay alpha",
        min_value=0.15,
        max_value=0.80,
        value=0.45,
        step=0.05,
    )

    unet_threshold = artifacts.unet_threshold
    if method == "U-Net":
        unet_threshold = st.slider(
            "U-Net threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(artifacts.unet_threshold),
            step=0.05,
        )

    image_file = st.file_uploader("Input image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    mask_file = st.file_uploader("Ground-truth mask", type=["png", "jpg", "jpeg", "tif", "tiff"])
    run_clicked = st.button("Run segmentation", type="primary", use_container_width=True)

    st.markdown('<div class="panel-title">Artifacts</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="artifact-note">
U-Net device: {artifacts.device_label()}<br>
U-Net threshold: {artifacts.unet_threshold:.2f}<br>
Output directory: outputs
</div>
        """,
        unsafe_allow_html=True,
    )
    if artifacts.svm_error:
        st.warning(artifacts.svm_error)
    if artifacts.unet_error:
        st.warning(artifacts.unet_error)

if image_file is None:
    st.markdown('<div class="empty-state">No input image selected.</div>', unsafe_allow_html=True)
    st.stop()

image = read_rgb_upload(image_file)
ground_truth = read_mask_upload(mask_file, image.shape[:2]) if mask_file is not None else None

if not run_clicked:
    st.markdown('<div class="section-title">Original</div>', unsafe_allow_html=True)
    st.image(image, caption="Input image", use_container_width=True)
    st.stop()

run_artifacts = replace(artifacts, unet_threshold=float(unet_threshold))

try:
    with st.spinner(f"Running {method}..."):
        result = run_inference(
            image,
            method,
            run_artifacts,
            ground_truth,
            overlay_alpha=float(overlay_alpha),
        )
except ModelUnavailableError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Inference failed: {type(exc).__name__}: {exc}")
    st.stop()

st.markdown(
    f'<div class="run-caption">{result.method} finished in {result.seconds:.3f}s</div>',
    unsafe_allow_html=True,
)

if ground_truth is None:
    cols = st.columns(3)
    cols[0].metric("Building count", f"{result.metrics['n_pred']:.0f}")
    cols[1].metric("Area ratio", f"{result.metrics['area_ratio_pred']:.2f}%")
    cols[2].metric("Padded shape", f"{result.padded_shape[0]} x {result.padded_shape[1]}")
else:
    cols = st.columns(6)
    cols[0].metric("IoU", metric_value(result.metrics, "iou"))
    cols[1].metric("Dice/F1", metric_value(result.metrics, "dice"))
    cols[2].metric("Precision", metric_value(result.metrics, "precision"))
    cols[3].metric("Recall", metric_value(result.metrics, "recall"))
    cols[4].metric("Count error", metric_value(result.metrics, "count_err"))
    cols[5].metric("Area error", f"{result.metrics['area_abs_error']:.2f}%")

st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
view_cols = st.columns(3)
view_cols[0].image(image, caption="Original", use_container_width=True)
view_cols[1].image(result.mask, caption="Predicted Mask", use_container_width=True, clamp=True)
view_cols[2].image(result.overlay, caption="Overlay", use_container_width=True)

if result.error_overlay is not None:
    st.markdown('<div class="section-title">Error Overlay</div>', unsafe_allow_html=True)
    st.image(result.error_overlay, caption="green=TP, red=FP, blue=FN", use_container_width=True)
