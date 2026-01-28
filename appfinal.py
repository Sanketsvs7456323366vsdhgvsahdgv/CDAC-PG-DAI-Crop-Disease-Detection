import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import base64  # <--- ONLY NEW IMPORT NEEDED FOR SLIDESHOW

# =========================
# PAGE CONFIG (MUST BE FIRST st.* CALL)
# =========================
st.set_page_config(page_title="Plant Disease Classifier", layout="centered")

# ==============================================================================
#  START OF UI/DESIGN BLOCK (Added to support Slideshow & Visibility)
# ==============================================================================

# 1. HELPER: Load Image to Base64
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# 2. LOAD BACKGROUND IMAGES (Ensure these files exist!)
img1 = get_base64_of_bin_file("D:\IMP_File\C-Dac_Main\Project\Final_project\img_1.jpeg")
img2 = get_base64_of_bin_file("D:\IMP_File\C-Dac_Main\Project\Final_project\crop2.jpg")
img3 = get_base64_of_bin_file("D:\IMP_File\C-Dac_Main\Project\Final_project\crop3.jpg")

# 3. INJECT CSS (Slideshow + Black Text Visibility Fix)
if img1 and img2 and img3:
    st.markdown(
        f"""
        <style>
        /* A. SLIDESHOW ANIMATION */
        @keyframes slideShow {{
            0% {{ background-image: url('data:image/jpg;base64,{img1}'); }}
            33% {{ background-image: url('data:image/jpg;base64,{img2}'); }}
            66% {{ background-image: url('data:image/jpg;base64,{img3}'); }}
            100% {{ background-image: url('data:image/jpg;base64,{img1}'); }}
        }}
        
        .stApp {{
            animation-name: slideShow;
            animation-duration: 15s;
            animation-iteration-count: infinite;
            animation-timing-function: ease-in-out;
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* B. MAIN CARD: Force White Background */
        .block-container {{
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            margin-top: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        /* C. TEXT VISIBILITY (The Nuclear Fix) */
        /* Force ALL text inside the main card to be BLACK, ignoring Dark Mode */
        .block-container h1, .block-container h2, .block-container h3, 
        .block-container p, .block-container div, .block-container span, 
        .block-container li, .block-container label, .block-container small {{
            color: #000000 !important;
        }}
        
        /* D. DRAG & DROP & UPLOADER FIX */
        [data-testid="stFileUploader"] div,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] label {{
            color: #000000 !important;
        }}

        /* E. SIDEBAR FIX (Light Background + Black Text) */
        [data-testid="stSidebar"] {{
            background-color: #f0f2f6 !important;
            border-right: 1px solid #ccc;
        }}
        [data-testid="stSidebar"] * {{
            color: #000000 !important;
        }}
        
        /* F. SIDEBAR ARROW FIX (Make it Visible & Black) */
        [data-testid="stSidebarCollapsedControl"] {{
            visibility: visible !important;
            display: block !important;
            color: #000000 !important; 
            background-color: rgba(255, 255, 255, 0.8); 
            border-radius: 5px;
            z-index: 1000002;
        }}
        
        /* G. CLEANUP: Hide Header & Footer */
        [data-testid="stHeader"] {{ background-color: rgba(0,0,0,0); }}
        [data-testid="stToolbar"] {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 4. SIDEBAR CONTENT
with st.sidebar:
    st.title("🌿 AgroScan AI")
    st.markdown("---")
    st.subheader("📌 How to Use")
    st.markdown("""
    1. **Upload Image:** Select a clear leaf photo.
    2. **Analysis:** The AI scans for disease patterns.
    3. **Results:** View the predicted disease and confidence score.
    """)
    st.markdown("---")
    st.subheader("👥 Project Made By")
    st.markdown("**PG-DAI Group 5**")
    st.caption("CDAC Project © 2026")

# ==============================================================================
#  END OF UI/DESIGN BLOCK
#  (BELOW IS YOUR EXACT APP4.PY CODE - UNCHANGED)
# ==============================================================================

# PATHS (USE EITHER ABSOLUTE OR RELATIVE, NOT BOTH)
# Option A (recommended for your current setup): use absolute paths safely with raw strings. [web:28]
MODEL_PATH = r"D:\IMP_File\C-Dac_Main\Project\Final_project\final 3\plant_resnet50.keras"
LABELS_PATH = r"D:\IMP_File\C-Dac_Main\Project\Final_project\class_names.json"

# Option B (when deploying): put files in ./model/ and use:
# MODEL_DIR = "model"
# MODEL_PATH = os.path.join(MODEL_DIR, "plant_resnet50_retrained.keras")
# LABELS_PATH = os.path.join(MODEL_DIR, "class_names.json")

IMG_SIZE = (224, 224)

# =========================
# LOAD MODEL + LABELS
# =========================
@st.cache_resource
def load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.stop()

    if not os.path.exists(LABELS_PATH):
        st.error(f"❌ Labels file not found: {LABELS_PATH}")
        st.stop()

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    except Exception as e:
        st.error(f"❌ Error loading class names: {e}")
        st.stop()

    # Ensure class_names is a list
    if isinstance(class_names, dict):
        # If it's {"0":"label0", ...} convert to list in correct order
        keys = list(class_names.keys())
        if all(str(k).isdigit() for k in keys):
            inv = {int(k): v for k, v in class_names.items()}
            n = max(inv.keys()) + 1
            class_names = [inv[i] for i in range(n)]
        else:
            # If it's {"label":0, ...}
            inv = {int(v): k for k, v in class_names.items()}
            n = max(inv.keys()) + 1
            class_names = [inv[i] for i in range(n)]

    class_names = [str(x) for x in class_names]
    return model, class_names

model, class_names = load_model_and_labels()

# =========================
# UI
# =========================
st.title("🌿 Plant Disease Classifier")
st.markdown("Upload an image of a plant leaf to classify potential diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Please upload an image to get a prediction.")
    st.stop()

# Display uploaded image
image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded Image", use_container_width=True)

# Preprocess
img_array = np.array(image.resize(IMG_SIZE), dtype=np.float32)  # keep 0..255 for preprocess_input [web:32]
img_array = np.expand_dims(img_array, axis=0)                   # (1, 224, 224, 3)

# IMPORTANT: Use the preprocess that matches your training.
# For ResNet50 ImageNet-style preprocessing: [web:32]
processed_img = tf.keras.applications.resnet50.preprocess_input(img_array)

# Predict
with st.spinner("Classifying..."):
    preds = model.predict(processed_img, verbose=0)[0]

# Safety: label/model mismatch check
if len(preds) != len(class_names):
    st.error(
        f"❌ Mismatch: model outputs {len(preds)} classes, "
        f"but class_names has {len(class_names)} labels."
    )
    st.stop()

pred_idx = int(np.argmax(preds))
pred_name = class_names[pred_idx]
confidence = float(np.max(preds)) * 100

st.success(f"Prediction: **{pred_name}**")
st.write(f"Confidence: **{confidence:.2f}%**")

# Top-5
st.subheader("Top 5 Predictions")
top5 = np.argsort(preds)[::-1][:5]
for i in top5:
    st.write(f"- {class_names[int(i)]}: {float(preds[int(i)]) * 100:.2f}%")

st.markdown("---")
st.markdown("Developed using TensorFlow/Keras and Streamlit")