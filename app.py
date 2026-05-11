"""
=============================================================
  Automatic Image Annotation (AIA) Web App

  Built with Streamlit + PyTorch (ResNet50 pre-trained)
=============================================================
"""

import streamlit as st
from PIL import Image
import io
import os

# Import our custom modules
from modules.annotator import annotate_image
from modules.feature_extractor import extract_color_features, extract_texture_features
from modules.visualizer import plot_confidence_chart, plot_color_histogram

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="AIA — Image Annotation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main { background-color: #0d0f14; }
    .stApp { background-color: #0d0f14; }

    h1, h2, h3 {
        font-family: 'Space Mono', monospace !important;
        color: #e2e8f0 !important;
    }

    .title-block {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
        border: 1px solid #2d3748;
        border-left: 4px solid #00d4aa;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .title-block h1 { font-size: 2rem; margin: 0; color: #00d4aa !important; }
    .title-block p  { color: #718096; margin: 0.4rem 0 0 0; font-size: 0.95rem; }

    .annotation-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: border-color 0.2s;
    }
    .annotation-card:hover { border-color: #00d4aa; }
    .annotation-label { font-family: 'Space Mono', monospace; color: #e2e8f0; font-size: 0.9rem; }
    .annotation-score { font-family: 'Space Mono', monospace; color: #00d4aa; font-weight: 700; font-size: 1rem; }

    .progress-bar-bg {
        background: #2d3748;
        border-radius: 4px;
        height: 6px;
        width: 180px;
        margin-top: 4px;
    }
    .progress-bar-fill {
        background: linear-gradient(90deg, #00d4aa, #0099ff);
        border-radius: 4px;
        height: 6px;
    }

    .feature-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
        color: #9ca3af;
    }
    .feature-key { color: #60a5fa; }
    .feature-val { color: #34d399; }

    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-family: 'Space Mono', monospace;
        background: #003d30;
        color: #00d4aa;
        border: 1px solid #00d4aa44;
        margin: 2px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00d4aa, #0099ff);
        color: #0d0f14;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
        transition: opacity 0.2s;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.85; }

    .sidebar-info {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.8rem;
        color: #718096;
        margin-bottom: 1rem;
    }
    .sidebar-info b { color: #00d4aa; }

    div[data-testid="stMetricValue"] { color: #00d4aa !important; font-family: 'Space Mono', monospace !important; }
    div[data-testid="stMetricLabel"] { color: #718096 !important; }

    .stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #718096; font-family: 'Space Mono', monospace; font-size: 0.8rem; }
    .stTabs [aria-selected="true"] { color: #00d4aa !important; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sidebar-info'>
        <b>MODEL</b><br>ResNet-50 (ImageNet)<br><br>
        <b>TOP-K LABELS</b><br>Adjustable (1–20)<br><br>
        <b>FEATURES</b><br>Colour Histogram · LBP Texture<br><br>
        <b>PROJECT</b><br>CSET-499 · TEAM 75
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Number of annotations", min_value=3, max_value=20, value=8)
    confidence_threshold = st.slider("Min. confidence (%)", min_value=0, max_value=50, value=5)
    show_features = st.checkbox("Show low-level features", value=True)
    show_charts = st.checkbox("Show confidence chart", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#4a5568; font-family: Space Mono, monospace;'>
    AIA · Auto Image Annotation<br>
    Deep Learning Review Project<br>
    ResNet50 · PyTorch · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────
st.markdown("""
<div class='title-block'>
    <h1>🔬 Automatic Image Annotation</h1>
    <p>Upload any image — the ResNet-50 deep learning model will auto-annotate it with semantic labels,
    confidence scores, and low-level visual feature analysis.</p>
</div>
""", unsafe_allow_html=True)


# ─── Upload Area ─────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1.5], gap="large")

with col_upload:
    st.markdown("### 📁 Upload Image")
    uploaded_file = st.file_uploader(
        "Drag & drop or click to upload",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"📷 {uploaded_file.name}", use_container_width=True)

        # File info metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Width", f"{image.width}px")
        m2.metric("Height", f"{image.height}px")
        m3.metric("Mode", image.mode)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  RUN ANNOTATION")
    else:
        st.markdown("""
        <div style='background:#111827; border:2px dashed #2d3748; border-radius:12px;
                    padding:3rem; text-align:center; color:#4a5568;'>
            <div style='font-size:3rem;'>🖼️</div>
            <div style='font-family: Space Mono, monospace; font-size:0.85rem; margin-top:0.5rem;'>
                JPG · PNG · BMP · WEBP
            </div>
        </div>
        """, unsafe_allow_html=True)
        run_btn = False


# ─── Results ─────────────────────────────────────────────────
with col_result:
    st.markdown("### 🏷️ Annotation Results")

    if uploaded_file and run_btn:
        with st.spinner("Running ResNet-50 inference..."):
            # Run annotation
            annotations = annotate_image(image, top_k=top_k, min_confidence=confidence_threshold)

            # Extract features
            color_features = extract_color_features(image)
            texture_features = extract_texture_features(image)

        if annotations:
            # Top label badge
            top_label = annotations[0]['label']
            top_score = annotations[0]['score']
            st.markdown(f"""
            <div style='margin-bottom:1rem;'>
                <span style='color:#718096; font-size:0.8rem; font-family:Space Mono,monospace;'>TOP PREDICTION</span><br>
                <span style='font-size:1.4rem; font-family:Space Mono,monospace; color:#00d4aa;'>{top_label}</span>
                <span style='font-size:1rem; color:#4a5568;'> · {top_score:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

            # Tabs for results view
            tab1, tab2, tab3 = st.tabs(["📋 ALL LABELS", "📊 CHART", "🔍 FEATURES"])

            with tab1:
                for ann in annotations:
                    bar_width = int(ann['score'] * 1.8)  # scale to ~180px max
                    st.markdown(f"""
                    <div class='annotation-card'>
                        <div>
                            <div class='annotation-label'>{ann['label']}</div>
                            <div class='progress-bar-bg'>
                                <div class='progress-bar-fill' style='width:{bar_width}px'></div>
                            </div>
                        </div>
                        <div class='annotation-score'>{ann['score']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Keyword badges
                st.markdown("<br>**Semantic Keywords:**", unsafe_allow_html=True)
                badges = " ".join([f"<span class='badge'>{a['label'].split(',')[0]}</span>"
                                   for a in annotations[:12]])
                st.markdown(badges, unsafe_allow_html=True)

            with tab2:
                if show_charts:
                    fig = plot_confidence_chart(annotations)
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("Enable 'Show confidence chart' in sidebar.")

            with tab3:
                if show_features:
                    st.markdown("**🎨 Colour Features**")
                    color_html = "".join([
                        f"<span class='feature-key'>{k}</span>: <span class='feature-val'>{v}</span><br>"
                        for k, v in color_features.items()
                    ])
                    st.markdown(f"<div class='feature-box'>{color_html}</div>", unsafe_allow_html=True)

                    st.markdown("<br>**🌀 Texture Features**", unsafe_allow_html=True)
                    tex_html = "".join([
                        f"<span class='feature-key'>{k}</span>: <span class='feature-val'>{v}</span><br>"
                        for k, v in texture_features.items()
                    ])
                    st.markdown(f"<div class='feature-box'>{tex_html}</div>", unsafe_allow_html=True)

                    st.markdown("<br>**📊 Colour Histogram**", unsafe_allow_html=True)
                    hist_fig = plot_color_histogram(image)
                    st.pyplot(hist_fig, use_container_width=True)
                else:
                    st.info("Enable 'Show low-level features' in sidebar.")

        else:
            st.warning("No annotations found above the confidence threshold. Try lowering it.")

    elif not uploaded_file:
        st.markdown("""
        <div style='background:#111827; border:1px solid #1f2937; border-radius:12px;
                    padding:3rem; text-align:center; color:#4a5568; margin-top:1rem;'>
            <div style='font-size:2.5rem;'>⬅️</div>
            <div style='font-family:Space Mono,monospace; font-size:0.8rem; margin-top:0.5rem;'>
                Upload an image to see annotations
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#111827; border:1px solid #1f2937; border-radius:12px;
                    padding:3rem; text-align:center; color:#4a5568; margin-top:1rem;'>
            <div style='font-size:2.5rem;'>▶️</div>
            <div style='font-family:Space Mono,monospace; font-size:0.8rem; margin-top:0.5rem;'>
                Press RUN ANNOTATION to start
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-family:Space Mono,monospace; font-size:0.72rem; color:#2d3748; padding:1rem;'>
    AIA WEB APP · CSET-499 CAPSTONE · TEAM 75 · ResNet-50 Pre-trained on ImageNet-1K
</div>
""", unsafe_allow_html=True)
