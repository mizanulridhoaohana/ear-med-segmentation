import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="🧩 Ear Diagnose System", layout="wide")
st.title("🧩 Ear Diagnose System: Perforation severity level")

# Custom CSS: make page wider and add border/outline for displayed images (sebelum & sesudah)
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1400px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    [data-testid="stImage"] > img {
        border: 3px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }
    .stMarkdown div[style*="background-color"] {
        padding: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Load ONNX model ---
@st.cache_resource
def load_onnx_model(model_path):
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

onnx_model_path = "./models/v2-updraded_models.onnx"
ort_session = load_onnx_model(onnx_model_path)

# --- Parameters ---
IMG_SIZE = (320, 320)
NUM_CLASSES = 3
cmap = np.array([
    [25, 25, 25],     # Background
    [51, 204, 51],    # Label 1
    [150, 0, 200],    # Label 2
], dtype=np.uint8)

# --- Session state ---
if "processed" not in st.session_state:
    st.session_state.processed = False
if "img" not in st.session_state:
    st.session_state.img = None

# --- Upload Section (Tengah Halaman) ---
st.markdown(
    """
    <div style="text-align:center; margin-top:20px; margin-bottom:30px;">
        <h3>📤 Upload Image untuk Segmentasi</h3>
        <p style="color:gray;">Unggah gambar telinga (format .jpg, .jpeg, atau .png)</p>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# --- Reset otomatis ketika upload baru ---
if uploaded_file is not None:
    if "last_uploaded_file" not in st.session_state or uploaded_file.name != st.session_state.last_uploaded_file:
        st.session_state.processed = False
        st.session_state.pred_mask = None
        st.session_state.color_mask = None
        st.session_state.overlay = None
        st.session_state.stats = None
        st.session_state.last_uploaded_file = uploaded_file.name

# --- Main Flow ---
if uploaded_file is not None:
    # Load dan resize gambar
    pil_img = Image.open(uploaded_file).convert("RGB")
    img = np.array(pil_img)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized.astype(np.float32), axis=0)
    st.session_state.img = img_resized

    # --- Preview sebelum diproses ---
    if not st.session_state.processed:
        st.subheader("🖼️ Preview Gambar (320×320)")
        st.image(img_resized, caption="Gambar yang sudah di-resize", use_container_width=True, output_format="PNG")

        # Tombol process
        col = st.columns([1, 1, 1])
        with col[1]:
            if st.button("🚀 Process Image", use_container_width=True):
                with st.spinner("Processing image..."):
                    # Prediksi segmentasi
                    input_name = ort_session.get_inputs()[0].name
                    output_name = ort_session.get_outputs()[0].name
                    pred = ort_session.run([output_name], {input_name: img_input})[0]
                    pred_mask = np.argmax(pred[0], axis=-1)
                    color_mask = cmap[pred_mask]
                    overlay = cv2.addWeighted(img_normalized, 0.7, color_mask / 255.0, 0.3, 0)

                    # Hitung statistik
                    label_1_area = np.sum(pred_mask == 1)
                    label_2_area = np.sum(pred_mask == 2)
                    label_0_area = np.sum(pred_mask == 0)
                    ratio = (label_2_area / label_1_area * 100) if label_1_area > 0 else 0

                    st.session_state.pred_mask = pred_mask
                    st.session_state.color_mask = color_mask
                    st.session_state.overlay = overlay
                    st.session_state.stats = {
                        "label_0_area": label_0_area,
                        "label_1_area": label_1_area,
                        "label_2_area": label_2_area,
                        "ratio": ratio
                    }
                    st.session_state.processed = True
                    st.rerun()

    # --- Setelah proses ---
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🧠 Sebelum Segmentasi")
            st.image(st.session_state.img, use_container_width=True)
        with col2:
            st.subheader("🔍 Sesudah Segmentasi (Overlay)")
            st.image(st.session_state.overlay, use_container_width=True)

        # --- Card Result ---
        st.markdown("---")
        st.markdown("### 📊 Hasil Segmentasi")

        stats = st.session_state.stats
        st.markdown(
            f"""
            <div style="display: flex; gap: 20px; flex-wrap: wrap; justify-content:center;">
                <div style="flex: 1; min-width:250px; background-color: #1e1e1e; padding: 20px; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
                    <h4>🟩 Gendang Telinga</h4>
                    <h2 style="color:#33CC33;">{stats['label_1_area']:,} px</h2>
                </div>
                <div style="flex: 1; min-width:250px; background-color: #1e1e1e; padding: 20px; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
                    <h4>🟪 Lubang Telinga</h4>
                    <h2 style="color:#9900CC;">{stats['label_2_area']:,} px</h2>
                </div>
                <div style="flex: 1; min-width:250px; background-color: #212e46; padding: 20px; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.4);">
                    <h4>💥 Persentase Kerusakan</h4>
                    <h1 style="color:#FF5555;">{stats['ratio']:.2f}%</h1>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col = st.columns([1, 1, 1])
        with col[1]:
            if st.button("🔄 Upload Gambar Baru", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
else:
    st.info("⬆️ Silakan upload gambar terlebih dahulu untuk melihat preview dan memproses segmentasi.")
