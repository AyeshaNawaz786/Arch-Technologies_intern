"""
╔════════════════════════════════════════════════════════════════════════════════╗
║           🎯 MNIST DIGIT RECOGNITION — STREAMLIT DEPLOYMENT APP                ║
║                                                                                ║
║  Loads the CNN model + artifacts produced by:                                 ║
║  TASK2_MNIST_Advanced_Colab.ipynb                                              ║
║                                                                                ║
║  Expects this folder structure:                                               ║
║    mnist_digit_project/                                                       ║
║      ├── app.py                     (this file)                              ║
║      └── models/                                                              ║
║          ├── mnist_cnn_model.h5                                               ║
║          ├── training_history.pkl                                             ║
║          ├── final_metrics.pkl                                                ║
║          └── model_comparison.csv                                             ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from PIL import Image, ImageOps

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import tensorflow as tf
from tensorflow import keras

import warnings
warnings.filterwarnings('ignore')

# Optional: drawable canvas (pip install streamlit-drawable-canvas)
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🎯 MNIST Digit Recognition",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════════════════
# CSS STYLING
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 50%, #16213e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1a2e 0%, #1a2847 100%);
    }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-weight: 700; }
    .stMarkdown { color: rgba(255,255,255,0.92) !important; }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; padding: 10px 26px; border-radius: 8px;
        font-weight: 600; transition: all .25s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 22px rgba(102,126,234,.45); }

    .header-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 34px; border-radius: 14px; text-align: center;
        margin-bottom: 26px; box-shadow: 0 8px 28px rgba(0,0,0,.3);
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(102,126,234,.15) 0%, rgba(118,75,162,.15) 100%);
        padding: 20px; border-radius: 12px; border: 1px solid rgba(102,126,234,.3);
        text-align: center; color: white;
    }
    .metric-card h4 { margin: 0; font-size: 13px; color: rgba(255,255,255,.7); }
    .metric-card p { margin: 8px 0 0 0; font-size: 28px; font-weight: 700; color: #8b9cf7; }

    .result-box {
        background: linear-gradient(135deg, rgba(34,197,94,.15) 0%, rgba(56,239,125,.15) 100%);
        border-left: 4px solid #22c55e; padding: 22px; border-radius: 10px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# LOAD MODEL & ARTIFACTS
# ════════════════════════════════════════════════════════════════════════════════

MODEL_PATH = 'models/mnist_cnn_model.h5'
HISTORY_PATH = 'models/training_history.pkl'
METRICS_PATH = 'models/final_metrics.pkl'
COMPARISON_PATH = 'models/model_comparison.csv'

@st.cache_resource
def load_cnn_model():
    if os.path.exists(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH)
    return None

@st.cache_data
def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_comparison():
    if os.path.exists(COMPARISON_PATH):
        return pd.read_csv(COMPARISON_PATH)
    return None

model = load_cnn_model()
history = load_history()
metrics = load_metrics()
comparison_df = load_comparison()

# ════════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def preprocess_image_array(img_array_28x28):
    """Takes a 28x28 grayscale numpy array (0-255) and returns model-ready input."""
    arr = img_array_28x28.astype('float32') / 255.0
    return arr.reshape(1, 28, 28, 1)

def predict_digit(img_array_28x28):
    x = preprocess_image_array(img_array_28x28)
    probs = model.predict(x, verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(probs[digit])
    return digit, confidence, probs

def pil_to_mnist_format(pil_image):
    """Convert an arbitrary uploaded/drawn PIL image into MNIST-style 28x28 grayscale, digit=white on black."""
    img = pil_image.convert('L')
    # If background looks light (mostly white), invert so digit is white-on-black like MNIST
    arr = np.array(img)
    if arr.mean() > 127:
        img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    return np.array(img)

def probability_bar_chart(probs):
    fig = go.Figure(data=[go.Bar(
        x=list(range(10)), y=probs,
        marker=dict(color=probs, colorscale='Viridis', showscale=False),
        text=[f"{p*100:.1f}%" for p in probs], textposition='outside'
    )])
    fig.update_layout(title="Prediction Confidence per Digit", xaxis_title="Digit",
                       yaxis_title="Probability", height=320, showlegend=False,
                       plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                       font=dict(color='white'))
    return fig

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0;">
        <h1 style="font-size:30px; margin:0;">🎯</h1>
        <h2 style="font-size:19px; margin:8px 0 4px 0;">MNIST Recognizer</h2>
        <p style="color:rgba(255,255,255,.55); font-size:12px; margin:0;">CNN-powered digit classifier</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigate", ["🏠 Home", "✏️ Draw & Predict", "📤 Upload Image",
                                  "📊 Model Performance", "⚖️ Model Comparison"],
                     label_visibility="collapsed")
    st.divider()

    if model is None:
        st.error("⚠️ Model not found.\n\nPlace `mnist_cnn_model.h5` inside the `models/` folder.")
    else:
        st.success("✅ Model loaded")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.markdown("""
    <div class="header-box">
        <h1 style="margin:0; font-size:38px;">🎯 MNIST Digit Recognition</h1>
        <p style="margin:10px 0 0 0; font-size:17px; opacity:.92;">
            Deep Learning (CNN) — Handwritten Digit Classifier
        </p>
    </div>
    """, unsafe_allow_html=True)

    acc = metrics['accuracy']*100 if metrics else 99.0
    n_params = f"{metrics['total_params']:,}" if metrics else "—"

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Test Accuracy", "Architecture", "Parameters", "Classes"],
        [f"{acc:.2f}%", "CNN (3 blocks)", n_params, "10 (0–9)"]
    ):
        with col:
            st.markdown(f'<div class="metric-card"><h4>{label}</h4><p>{value}</p></div>', unsafe_allow_html=True)

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### ✏️ Draw & Predict")
        st.write("Draw a digit on a live canvas and get an instant prediction with a full confidence breakdown.")
    with c2:
        st.markdown("#### 📤 Upload Image")
        st.write("Upload one or more digit images (PNG/JPG) for batch classification.")
    with c3:
        st.markdown("#### 📊 Model Insights")
        st.write("Explore training curves, per-digit performance, and a comparison against classical ML models.")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: DRAW & PREDICT
# ════════════════════════════════════════════════════════════════════════════════

elif page == "✏️ Draw & Predict":
    st.markdown("## ✏️ Draw a Digit")

    if model is None:
        st.error("Model not loaded — cannot predict. Add `models/mnist_cnn_model.h5` and reload.")
    elif not CANVAS_AVAILABLE:
        st.warning("The drawing canvas requires an extra package.\n\nRun:\n```\npip install streamlit-drawable-canvas\n```\nthen restart the app. Meanwhile, use **📤 Upload Image** instead.")
    else:
        col1, col2 = st.columns([1.3, 1])
        with col1:
            st.caption("Draw a single digit (0–9), centered, using a thick stroke.")
            canvas_result = st_canvas(
                fill_color="#000000",
                stroke_width=18,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280, width=280,
                drawing_mode="freedraw",
                key="canvas"
            )
            b1, b2 = st.columns(2)
            predict_clicked = b1.button("🔮 Predict Digit", use_container_width=True)
            b2.button("🗑️ Clear (use canvas trash icon)", use_container_width=True, disabled=True)

        with col2:
            st.markdown("### Result")
            if predict_clicked and canvas_result.image_data is not None:
                pil_img = Image.fromarray(canvas_result.image_data.astype('uint8'), mode='RGBA').convert('L')
                arr = np.array(pil_img)
                if arr.max() == 0:
                    st.warning("Canvas is empty — please draw a digit first.")
                else:
                    img28 = np.array(Image.fromarray(arr).resize((28, 28), Image.Resampling.LANCZOS))
                    digit, confidence, probs = predict_digit(img28)

                    st.markdown(f"""
                    <div class="result-box">
                        <h1 style="margin:0; font-size:52px;">{digit}</h1>
                        <p style="margin:8px 0 0 0; font-size:16px;">Confidence: {confidence*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(probability_bar_chart(probs), use_container_width=True)
            else:
                st.info("Draw a digit and click **Predict Digit**.")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD IMAGE
# ════════════════════════════════════════════════════════════════════════════════

elif page == "📤 Upload Image":
    st.markdown("## 📤 Upload Digit Image(s)")

    if model is None:
        st.error("Model not loaded — cannot predict.")
    else:
        uploaded_files = st.file_uploader("Upload one or more images (PNG/JPG)",
                                           type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

        if uploaded_files:
            if st.button("🔮 Predict All", use_container_width=True):
                results = []
                cols = st.columns(4)
                for i, file in enumerate(uploaded_files):
                    pil_img = Image.open(file)
                    img28 = pil_to_mnist_format(pil_img)
                    digit, confidence, probs = predict_digit(img28)
                    results.append({'File': file.name, 'Predicted Digit': digit,
                                     'Confidence': f"{confidence*100:.1f}%"})
                    with cols[i % 4]:
                        st.image(img28, caption=f"Pred: {digit} ({confidence*100:.1f}%)", width=120)

                st.divider()
                st.markdown("### 📋 Results Summary")
                st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.info("Upload digit images to classify them.")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════

elif page == "📊 Model Performance":
    st.markdown("## 📊 Model Performance Dashboard")

    if metrics is None:
        st.warning("`final_metrics.pkl` not found — showing placeholder values. Run the Colab notebook and copy the file into `models/`.")
        acc, prec, rec, f1 = 0.992, 0.991, 0.990, 0.991
    else:
        acc, prec, rec, f1 = metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip([c1, c2, c3, c4], ["Accuracy", "Precision", "Recall", "F1-Score"],
                                [acc, prec, rec, f1]):
        with col:
            st.markdown(f'<div class="metric-card"><h4>{label}</h4><p>{val*100:.2f}%</p></div>', unsafe_allow_html=True)

    st.divider()

    if history:
        epochs = list(range(1, len(history['loss']) + 1))
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'Accuracy'))
        fig.add_trace(go.Scatter(x=epochs, y=history['loss'], name='Train Loss', mode='lines+markers'), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss', mode='lines+markers'), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history['accuracy'], name='Train Acc', mode='lines+markers'), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=history['val_accuracy'], name='Val Acc', mode='lines+markers'), row=1, col=2)
        fig.update_layout(height=420, title_text="Training History (from Colab run)",
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           font=dict(color='white'), legend=dict(orientation='h', y=-0.25))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("`training_history.pkl` not found — training curves unavailable. Copy it from the Colab `models/` folder.")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════════

elif page == "⚖️ Model Comparison":
    st.markdown("## ⚖️ CNN vs Classical ML Models")

    if comparison_df is None:
        st.warning("`model_comparison.csv` not found. Copy it from the Colab `models/` folder to see this comparison.")
    else:
        fig = go.Figure()
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            fig.add_trace(go.Bar(x=comparison_df['Model'], y=comparison_df[metric], name=metric))
        fig.update_layout(barmode='group', height=460, title_text="Model Comparison",
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           font=dict(color='white'), legend=dict(orientation='h', y=-0.25))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(comparison_df, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("""
<div style="text-align:center; padding:16px 0; color:rgba(255,255,255,.45); font-size:12px;">
    🎯 <strong>MNIST Digit Recognition</strong> · CNN + Streamlit · Task 2, Machine Learning Internship
</div>
""", unsafe_allow_html=True)
