"""
SIH26038 — Explainable AI-Based Diabetic Retinopathy Screening
Web Application & Screening Report Interface
"""

import os
import io
import base64
import traceback
from PIL import Image
import numpy as np
import streamlit as st

from src.inference import run_pipeline, generate_static_report_image, GRADE_LABELS

# Streamlit Page Config
st.set_page_config(
    page_title="SIH26038 — DR Screening Report",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Dark Near-Black/Green Medical AI Theme
STYLING_CSS = """<style>
/* Main Container & Background */
.stApp {
    background-color: #071613 !important;
    color: #F0F5F3 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* Hide Streamlit Header & Footer */
header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 860px !important;
}

/* Main Report Card */
.report-card {
    background-color: #0B1D1A;
    border: 1px solid #133D35;
    border-left: 5px solid #10B981;
    border-radius: 12px;
    padding: 32px 36px;
    margin-top: 10px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

.report-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #F0F5F3;
    letter-spacing: -0.01em;
    margin-bottom: 4px;
}

.report-subtitle {
    font-size: 1.05rem;
    font-weight: 500;
    color: #96B4AA;
    margin-bottom: 24px;
}

/* Metrics Row */
.metrics-grid {
    display: flex;
    justify-content: space-between;
    gap: 20px;
    margin-bottom: 24px;
}

.metric-box {
    display: flex;
    flex-direction: column;
    flex: 1;
}

.metric-label {
    font-size: 0.85rem;
    color: #96B4AA;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #F0F5F3;
}

/* Confidence Progress Bar */
.confidence-container {
    margin-bottom: 28px;
}

.confidence-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.confidence-label {
    font-size: 0.85rem;
    color: #96B4AA;
}

.confidence-pct {
    font-size: 1.05rem;
    font-weight: 700;
    color: #10B981;
}

.progress-track {
    width: 100%;
    height: 8px;
    background-color: #142F2A;
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid #133D35;
}

.progress-fill {
    height: 100%;
    background-color: #10B981;
    border-radius: 4px;
}

/* 3 Image Panels Grid */
.images-grid {
    display: flex;
    justify-content: space-between;
    gap: 20px;
    margin-bottom: 28px;
}

.image-panel {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
}

.image-panel-title {
    font-size: 0.85rem;
    font-weight: 500;
    color: #96B4AA;
    margin-bottom: 12px;
    text-align: center;
    height: 24px;
}

.retinal-img {
    width: 190px;
    height: 190px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #10B981;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
}

/* Referral Section */
.referral-box {
    margin-bottom: 24px;
    font-size: 1.25rem;
    font-weight: 700;
}

.referral-yes {
    color: #EF4444;
}

.referral-no {
    color: #10B981;
}

/* AI Explanation Box */
.explanation-section {
    margin-bottom: 28px;
}

.explanation-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #F0F5F3;
    margin-bottom: 8px;
}

.explanation-body {
    font-size: 0.9rem;
    line-height: 1.5;
    color: #96B4AA;
}

/* Medical Disclaimer */
.disclaimer-section {
    font-size: 0.8rem;
    line-height: 1.4;
    color: #F59E0B;
    border-top: 1px solid #133D35;
    padding-top: 16px;
    margin-top: 8px;
}

/* Streamlit Button Overrides */
.stButton > button {
    background-color: #10B981 !important;
    color: #071613 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 24px !important;
}

.stButton > button:hover {
    background-color: #059669 !important;
    color: #FFFFFF !important;
}
</style>"""

if hasattr(st, "html"):
    st.html(STYLING_CSS)
else:
    st.markdown(STYLING_CSS, unsafe_allow_html=True)


def numpy_to_base64(img_np: np.ndarray) -> str:
    """Helper to convert RGB numpy array to base64 PNG string for HTML display."""
    pil_img = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def render_report_card(results: dict):
    """Renders the HTML report card matching the target UI design."""
    orig_b64 = numpy_to_base64(results["circ_original_np"])
    gradcam_b64 = numpy_to_base64(results["circ_gradcam_np"])
    vessel_b64 = numpy_to_base64(results["circ_vessel_np"])

    conf_float = results["confidence"]
    conf_pct_str = results["confidence_pct"]
    referral_cls = "referral-yes" if results["referable"] else "referral-no"

    # Fully unindented HTML string so markdown parser never treats any line as a code block
    card_html = f"""<div class="report-card">
<div class="report-title">SIH26038 — Explainable DR Screening Report</div>
<div class="report-subtitle">Screening result</div>
<div class="metrics-grid">
<div class="metric-box">
<span class="metric-label">Image quality</span>
<span class="metric-value">{results['image_quality']}</span>
</div>
<div class="metric-box">
<span class="metric-label">Prediction</span>
<span class="metric-value">{results['prediction_label']}</span>
</div>
</div>
<div class="confidence-container">
<div class="confidence-header">
<span class="confidence-label">Confidence</span>
<span class="confidence-pct">{conf_pct_str}</span>
</div>
<div class="progress-track">
<div class="progress-fill" style="width: {conf_float * 100}%;"></div>
</div>
</div>
<div class="images-grid">
<div class="image-panel">
<div class="image-panel-title">Original</div>
<img src="data:image/png;base64,{orig_b64}" class="retinal-img" alt="Original Fundus" />
</div>
<div class="image-panel">
<div class="image-panel-title">AI Attention (Grad-CAM)</div>
<img src="data:image/png;base64,{gradcam_b64}" class="retinal-img" alt="Grad-CAM" />
</div>
<div class="image-panel">
<div class="image-panel-title">Retinal Vessel Enhancement</div>
<img src="data:image/png;base64,{vessel_b64}" class="retinal-img" alt="Vessel Enhancement" />
</div>
</div>
<div class="referral-box">
Referral: <span class="{referral_cls}">{results['referral_str']}</span>
</div>
<div class="explanation-section">
<div class="explanation-title">AI Explanation</div>
<div class="explanation-body">{results['explanation']}</div>
</div>
<div class="disclaimer-section">
Not a medical diagnosis.<br>
Ophthalmic evaluation recommended.
</div>
</div>"""

    if hasattr(st, "html"):
        st.html(card_html)
    else:
        st.markdown(card_html, unsafe_allow_html=True)


def main():
    st.markdown("<h2 style='color: #F0F5F3; margin-bottom: 4px;'>Diabetic Retinopathy AI Screening</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #96B4AA; margin-bottom: 20px;'>Explainable Two-Stage Machine Learning Pipeline</p>", unsafe_allow_html=True)

    # Input Options Card
    col_input1, col_input2 = st.columns([1, 1])

    unseen_test_dir = r"D:\diabetic\dataset\test_images"
    sample_images = []
    if os.path.exists(unseen_test_dir):
        sample_images = [f.replace(".png", "") for f in os.listdir(unseen_test_dir) if f.endswith(".png")]

    selected_sample = None
    uploaded_file = None

    with col_input1:
        uploaded_file = st.file_uploader("Upload Fundus Image", type=["png", "jpg", "jpeg"])

    with col_input2:
        if sample_images:
            selected_sample = st.selectbox(
                "Or Select Unseen Test Image",
                options=["-- Select Test Image --"] + sample_images,
                index=0
            )

    image_to_process = None
    image_name = "uploaded_fundus"

    if uploaded_file is not None:
        try:
            image_to_process = Image.open(uploaded_file)
            image_name = os.path.splitext(uploaded_file.name)[0]
        except Exception as e:
            st.error("Failed to read uploaded image. Please provide a valid fundus image.")
            print(f"[Error] Failed reading uploaded file: {e}")

    elif selected_sample and selected_sample != "-- Select Test Image --":
        sample_path = os.path.join(unseen_test_dir, f"{selected_sample}.png")
        if os.path.exists(sample_path):
            try:
                image_to_process = Image.open(sample_path)
                image_name = selected_sample
            except Exception as e:
                st.error("Failed to load selected test image.")
                print(f"[Error] Failed loading sample {sample_path}: {e}")

    if image_to_process is not None:
        if st.button("Analyze Image & Generate Report", use_container_width=True):
            with st.spinner("Executing screening pipeline..."):
                try:
                    # Run backend inference pipeline
                    results = run_pipeline(image_to_process)

                    # Export static report image to D:\diabetic\outputs\unseen_demo\
                    export_filename = f"unseen_report_{image_name}.png"
                    export_path = generate_static_report_image(results, filename=export_filename)

                    st.session_state["latest_results"] = results
                    st.session_state["latest_export_path"] = export_path
                    st.session_state["latest_export_filename"] = export_filename

                except Exception as ex:
                    # Clean error display to user while preserving full traceback in log
                    st.error("An error occurred during inference. Please check terminal logs.")
                    print("[EXPLICIT LOG ERROR] Pipeline Inference Failed:")
                    traceback.print_exc()

    # Display Report if available in session state
    if "latest_results" in st.session_state:
        res = st.session_state["latest_results"]
        render_report_card(res)

        # Download / Export Section
        export_path = st.session_state.get("latest_export_path")
        export_filename = st.session_state.get("latest_export_filename", "screening_report.png")

        if export_path and os.path.exists(export_path):
            with open(export_path, "rb") as f:
                btn = st.download_button(
                    label="📥 Save / Export Report (PNG)",
                    data=f.read(),
                    file_name=export_filename,
                    mime="image/png"
                )
            st.caption(f"Static report saved to: `{export_path}`")


if __name__ == "__main__":
    main()
