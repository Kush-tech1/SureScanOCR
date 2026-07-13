"""
Meter Reading Pipeline — Streamlit App
Run with:  streamlit run app.py
"""

import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Meter Reading Pipeline",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
    <style>
        .stApp [data-testid="stHeader"] {
            background: rgba(255,255,255,0.9);
        }
    </style>
""", unsafe_allow_html=True)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* App */
    .stApp {
        background-color: #f8fafc;
        color: #111827;
    }

    /* Main content spacing */
    .block-container {
        padding-top: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] * {
        color: #111827 !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
    }

    p, label, span, div {
        color: #374151;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    div[data-testid="metric-container"] label {
        color: #6b7280 !important;
        font-size: 0.8rem;
    }

    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 1.4rem;
        font-weight: 700;
    }

    /* Stage cards */
    .stage-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }

    .stage-title {
        font-size: 0.82rem;
        color: #6b7280;
        margin-bottom: 4px;
    }

    .stage-value {
        font-size: 1.05rem;
        font-weight: 600;
        color: #111827;
    }

    .stage-conf {
        font-size: 0.78rem;
        color: #2563eb;
        margin-top: 2px;
    }

    .stage-time {
        font-size: 0.72rem;
        color: #9ca3af;
        margin-top: 4px;
    }

    /* Reading highlight */
    .reading-box {
        background: linear-gradient(135deg, #eff6ff, #ffffff);
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 20px 28px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(37,99,235,0.08);
    }

    .reading-big {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1d4ed8;
        letter-spacing: 2px;
    }

    .reading-sub {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 4px;
    }

    /* Timing */
    .timing-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 5px 0;
    }

    .timing-label {
        font-size: 0.78rem;
        color: #6b7280;
        width: 160px;
        flex-shrink: 0;
    }

    .timing-bar {
        height: 8px;
        border-radius: 4px;
        background: #2563eb;
    }

    .timing-sec {
        font-size: 0.78rem;
        color: #2563eb;
        width: 48px;
        text-align: right;
    }

    /* Confidence pills */
    .conf-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .conf-high { background: #dcfce7; color: #15803d; }
    .conf-mid  { background: #fef3c7; color: #b45309; }
    .conf-low  { background: #fee2e2; color: #b91c1c; }

    /* Divider */
    hr {
        border-color: #e5e7eb;
    }

    /* Upload */
    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border: 2px dashed #cbd5e1;
        border-radius: 14px;
        padding: 12px;
    }

    /* Info / alert box */
    div[data-testid="stAlert"] {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e3a8a;
        border-radius: 12px;
    }

    /* Slider */
    .stSlider > div[data-baseweb="slider"] {
        padding-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load pipeline (cached) ───────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_pipeline():
    from pipeline import MeterPipeline
    return MeterPipeline(save_crops=False)


def conf_pill(val):
    if val is None:
        return '<span class="conf-pill conf-low">N/A</span>'
    pct = val * 100
    cls = "conf-high" if pct >= 80 else ("conf-mid" if pct >= 50 else "conf-low")
    return f'<span class="conf-pill {cls}">{pct:.1f}%</span>'


def timing_bars(timings: dict):
    stages = [
        ("S1 — Detection",  "s1_detection"),
        ("S2A — Brand",     "s2a_brand"),
        ("S2B — OCR",       "s2b_ocr"),
        ("S3 — Dewarp",     "s3_dewarp"),
        ("S4 — Digits",     "s4_digits"),
    ]
    total = timings.get("total", 1) or 1
    html = ""
    for label, key in stages:
        val = timings.get(key, 0) or 0
        pct = min(int(val / total * 100), 100)
        html += f"""
        <div class="timing-row">
            <span class="timing-label">{label}</span>
            <div style="flex:1; background:#2e3250; border-radius:4px; height:8px;">
                <div class="timing-bar" style="width:{pct}%;"></div>
            </div>
            <span class="timing-sec">{val:.2f}s</span>
        </div>"""
    return html


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Meter Pipeline")
    st.markdown("---")

    st.markdown("### Thresholds")
    detect_thr = st.slider("Detection threshold",  0.1, 1.0, 0.30, 0.05)
    dewarp_thr = st.slider("Dewarp threshold",     0.1, 1.0, 0.50, 0.05)
    digit_thr  = st.slider("Digit threshold",      0.1, 1.0, 0.70, 0.05)

    st.markdown("---")
    st.markdown("### Debug")
    show_crops = st.checkbox("Show intermediate crops", value=True)

    st.markdown("---")
    st.caption("Models loaded from working directory.\nEdit `CFG` in `pipeline.py` to change paths.")


# ── Header ───────────────────────────────────────────────────
st.markdown("# ⚡ Meter Reading Pipeline")
st.markdown("Upload one or more meter images to run the full 4-stage inference pipeline.")
st.markdown("---")

# ── File upload ──────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload meter image(s)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
)

if not uploaded:
    st.info("👆 Upload at least one image to get started.")
    st.stop()

# ── Load pipeline ────────────────────────────────────────────
try:
    pipeline = load_pipeline()
    # Apply sidebar thresholds dynamically
    from pipeline import CFG
    CFG["detect_thr"] = detect_thr
    CFG["dewarp_thr"] = dewarp_thr
    CFG["digit_thr"]  = digit_thr
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

# ── Process each image ───────────────────────────────────────
for uploaded_file in uploaded:
    img_pil = Image.open(uploaded_file).convert("RGB")

    st.markdown(f"## 🖼 `{uploaded_file.name}`")

    # Run pipeline
    with st.spinner(f"Running pipeline on {uploaded_file.name}…"):
        result = pipeline.run(img_pil, img_name=uploaded_file.name)

    s1 = result["_s1"]
    s3 = result["_s3"]
    s4 = result["_s4"]
    timings = result.get("timings", {})

    # ── Top row: original image + reading ───────────────────
    col_img, col_read = st.columns([1, 1])

    with col_img:
        st.markdown("**Original Image**")
        st.image(img_pil, use_container_width=True)

    with col_read:
        # Big reading box
        reading_val = result["reading"] or "N/A"
        st.markdown(f"""
        <div class="reading-box" style="margin-top:8px;">
            <div class="reading-big">{reading_val}</div>
            <div class="reading-sub">Meter Reading</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Key metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Brand",  result["brand"] or "N/A",
                  f"{result['brand_conf']*100:.1f}%" if result["brand_conf"] else None)
        m2.metric("Serial", result["serial_number"] or "N/A",
                  f"{result['serial_conf']*100:.1f}%" if result["serial_conf"] else None)
        m3.metric("Total Time", f"{result['elapsed_sec']}s")

        st.markdown("<br>", unsafe_allow_html=True)

        # Timing breakdown
        st.markdown("**⏱ Stage Timings**")
        st.markdown(timing_bars(timings), unsafe_allow_html=True)

    st.markdown("---")

    # ── Stage detail cards ───────────────────────────────────
    st.markdown("### Stage Details")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        det_score = result["meter_score"]
        sn_score  = result["sn_score"]
        st.markdown(f"""
        <div class="stage-card">
            <div class="stage-title">Stage 1 — RF-DETR Detection</div>
            <div class="stage-value">Meter: {conf_pill(det_score)}</div>
            <div class="stage-conf">SN: {conf_pill(sn_score)}</div>
            <div class="stage-time">⏱ {timings.get('s1_detection','N/A')}s</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="stage-card">
            <div class="stage-title">Stage 2A — Brand</div>
            <div class="stage-value">{result['brand'] or 'N/A'}</div>
            <div class="stage-conf">{conf_pill(result['brand_conf'])}</div>
            <div class="stage-time">⏱ {timings.get('s2a_brand','N/A')}s</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="stage-card">
            <div class="stage-title">Stage 2B — Serial OCR</div>
            <div class="stage-value">{result['serial_number'] or 'N/A'}</div>
            <div class="stage-conf">{conf_pill(result['serial_conf'])}</div>
            <div class="stage-time">⏱ {timings.get('s2b_ocr','N/A')}s</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        dw = "✅ Dewarped" if s3["keypoints"] is not None else "⚠️ Raw fallback"
        st.markdown(f"""
        <div class="stage-card">
            <div class="stage-title">Stage 3 — Dewarp</div>
            <div class="stage-value">{dw}</div>
            <div class="stage-conf">{conf_pill(result['dewarp_score'])}</div>
            <div class="stage-time">⏱ {timings.get('s3_dewarp','N/A')}s</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Digit tokens ────────────────────────────────────────
    if s4.get("tokens"):
        st.markdown("### Digit Tokens (Stage 4)")
        cols = st.columns(min(len(s4["tokens"]), 12))
        for i, tok in enumerate(s4["tokens"]):
            with cols[i % len(cols)]:
                color = "#4caf50" if tok["type"] == "digit" else ("#ff9800" if tok["type"] == "punct" else "#5b9bd5")
                st.markdown(f"""
                <div style="background:#1c1f2e; border:1px solid {color}; border-radius:8px;
                            padding:8px; text-align:center; margin-bottom:6px;">
                    <div style="font-size:1.6rem; font-weight:800; color:{color};">{tok['char']}</div>
                    <div style="font-size:0.68rem; color:#8b8fa8;">{tok['conf']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Intermediate crops ──────────────────────────────────
    if show_crops:
        st.markdown("### 🔍 Intermediate Crops")

        crop_cols = st.columns(4)
        crops = [
            ("S1 — Meter Body",    s1.get("meter_crop")),
            ("S1 — Serial Number", s1.get("sn_crop")),
            ("S3 — Dewarped",      s3.get("dewarped_crop")),
        ]

        for i, (label, crop_img) in enumerate(crops):
            with crop_cols[i]:
                st.markdown(f"**{label}**")
                if crop_img is not None:
                    st.image(crop_img, use_container_width=True)
                else:
                    st.markdown(
                        '<div style="background:#1c1f2e;border:1px solid #2e3250;'
                        'border-radius:8px;padding:20px;text-align:center;color:#6b7a8d;">'
                        'Not detected</div>',
                        unsafe_allow_html=True
                    )

    st.markdown("<br><br>", unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.caption("Instinct 4.0 — Meter Reading Pipeline · RF-DETR + EfficientNet_B0 + PaddleOCR (SVTR_HGNet) + Keypoint RCNN")