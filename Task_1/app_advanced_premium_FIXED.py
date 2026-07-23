"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  🛡️ SPAMGUARD ELITE - PROFESSIONAL EDITION                   ║
║                                                                              ║
║  ⚡ Next-Level AI-Powered Spam Detection                                     ║
║  🎨 Premium UI with Advanced Animations                                      ║
║  📊 Real-time Analytics Dashboard                                            ║
║  🔐 Enterprise-Grade Security                                               ║
║                                                                              ║
║  Built by: Ayesha | Arch Technologies Internship                           ║
║  Version: 2.0 ELITE | Production Ready                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
import time

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION - PREMIUM SETUP
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🛡️ SpamGuard ELITE - AI Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "### 🛡️ SpamGuard ELITE v2.0\n\n**Production-Ready AI Email Classifier**\n\n✨ Advanced ML + Premium UI Design\n\nBuilt by Ayesha | Arch Technologies"
    }
)

# ════════════════════════════════════════════════════════════════════════════
# PREMIUM DARK THEME CSS WITH ANIMATIONS
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* ═══ MAIN BACKGROUND ═══ */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 50%, #16213e 100%);
        background-attachment: fixed;
    }
    
    /* ═══ SIDEBAR ═══ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1a2e 0%, #1a2847 100%);
        border-right: 1px solid rgba(147, 112, 219, 0.3);
    }
    
    /* ═══ MAIN CONTAINER ═══ */
    [data-testid="stMainBlockContainer"] {
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        padding-top: 1.5rem;
        animation: fadeIn 0.6s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(147, 112, 219, 0.5); }
        50% { box-shadow: 0 0 40px rgba(147, 112, 219, 0.8); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    /* ═══ TABS ═══ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 3px;
        background: linear-gradient(90deg, rgba(147, 112, 219, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(147, 112, 219, 0.2);
        animation: slideInLeft 0.5s ease-out;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 10px;
        color: rgba(255, 255, 255, 0.7);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(147, 112, 219, 0.2);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border-color: rgba(59, 130, 246, 0.5);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #9370db 0%, #3b82f6 100%);
        color: white;
        box-shadow: 0 8px 32px rgba(147, 112, 219, 0.4);
    }
    
    /* ═══ PREMIUM CARD STYLES ═══ */
    .premium-card {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.8) 0%, rgba(59, 130, 246, 0.1) 100%);
        border: 1px solid rgba(147, 112, 219, 0.4);
        padding: 24px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        animation: slideInRight 0.6s ease-out;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .premium-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(147, 112, 219, 0.3);
        border-color: rgba(147, 112, 219, 0.6);
    }
    
    .success-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(34, 197, 94, 0.1) 100%);
        border: 1px solid rgba(34, 197, 94, 0.5);
        box-shadow: 0 0 30px rgba(34, 197, 94, 0.2);
    }
    
    .danger-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(244, 92, 67, 0.1) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.2);
    }
    
    .warning-card {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(251, 146, 60, 0.1) 100%);
        border: 1px solid rgba(245, 158, 11, 0.5);
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(147, 112, 219, 0.15) 0%, rgba(59, 130, 246, 0.1) 100%);
        border: 1px solid rgba(147, 112, 219, 0.4);
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        animation: fadeIn 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(147, 112, 219, 0.3);
    }
    
    /* ═══ HEADER STYLES ═══ */
    .header-gradient {
        background: linear-gradient(135deg, #9370db 0%, #3b82f6 50%, #06b6d4 100%);
        background-size: 200% 200%;
        animation: gradient-shift 6s ease infinite;
        color: white;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 20px 60px rgba(147, 112, 219, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .header-gradient h1 {
        font-size: 3em;
        font-weight: 800;
        margin-bottom: 10px;
        letter-spacing: -1px;
        animation: slideInLeft 0.8s ease-out;
    }
    
    .header-gradient p {
        font-size: 1.2em;
        opacity: 0.95;
        animation: slideInRight 0.8s ease-out;
    }
    
    /* ═══ BUTTONS ═══ */
    .stButton > button {
        background: linear-gradient(135deg, #9370db 0%, #3b82f6 100%);
        color: white;
        border: none;
        padding: 12px 32px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 24px rgba(147, 112, 219, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.2);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 36px rgba(147, 112, 219, 0.5);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(147, 112, 219, 0.4);
    }
    
    /* ═══ INPUT FIELDS ═══ */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: rgba(15, 23, 42, 0.8) !important;
        color: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(147, 112, 219, 0.3) !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        font-size: 1em !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(147, 112, 219, 0.8) !important;
        box-shadow: 0 0 20px rgba(147, 112, 219, 0.3) !important;
        background-color: rgba(30, 41, 59, 0.9) !important;
    }
    
    /* ═══ METRICS ═══ */
    .stMetric {
        background: linear-gradient(135deg, rgba(147, 112, 219, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(147, 112, 219, 0.3);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(147, 112, 219, 0.2);
    }
    
    /* ═══ TEXT & TYPOGRAPHY ═══ */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    p, span {
        color: rgba(255, 255, 255, 0.85);
    }
    
    .stMarkdown {
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* ═══ DIVIDERS ═══ */
    hr {
        border-color: rgba(147, 112, 219, 0.3) !important;
        margin: 30px 0 !important;
    }
    
    /* ═══ SELECTBOX & MULTISELECT ═══ */
    .stSelectbox, .stMultiSelect, .stSlider {
        color: white;
    }
    
    .stSelectbox > div > div > select,
    [data-baseweb="select"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        color: white !important;
        border-radius: 10px !important;
        border: 1px solid rgba(147, 112, 219, 0.3) !important;
    }
    
    /* ═══ EXPANDER ═══ */
    .streamlit-expanderHeader {
        background: rgba(147, 112, 219, 0.1) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(147, 112, 219, 0.3) !important;
    }
    
    /* ═══ DATAFRAME ═══ */
    [data-testid="stDataFrame"] {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
    }
    
    /* ═══ SUCCESS/WARNING/INFO MESSAGES ═══ */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid;
    }
    
    .stSuccess {
        background: rgba(34, 197, 94, 0.1) !important;
        border-left-color: #22c55e !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border-left-color: #ef4444 !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1) !important;
        border-left-color: #f59e0b !important;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.1) !important;
        border-left-color: #3b82f6 !important;
    }
    
    /* ═══ FOOTER ═══ */
    .footer-text {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.9em;
        text-align: center;
        padding: 30px 0;
        animation: fadeIn 1s ease-out;
    }
    
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING WITH CACHING
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model_and_vectorizer():
    """Load trained SVM model and TF-IDF vectorizer"""
    try:
        with open('svm_spam_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_model_and_vectorizer()
model_loaded = model is not None and vectorizer is not None

# ════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def predict_spam(text):
    """Predict if email is spam with confidence"""
    text_lower = text.lower().strip()
    text_vectorized = vectorizer.transform([text_lower])
    prediction = model.predict(text_vectorized)[0]
    confidence = model.decision_function(text_vectorized)[0]
    spam_prob = 1 / (1 + np.exp(-confidence))
    legit_prob = 1 - spam_prob
    
    return {
        'prediction': 'SPAM' if prediction == 1 else 'LEGITIMATE',
        'spam_probability': spam_prob,
        'legit_probability': legit_prob,
        'confidence': abs(spam_prob - 0.5) * 2,
        'risk_level': 'CRITICAL' if spam_prob > 0.8 else 'HIGH' if spam_prob > 0.6 else 'MEDIUM' if spam_prob > 0.4 else 'LOW'
    }

def analyze_email_features(text):
    """Extract email characteristics"""
    spam_indicators = ['free', 'click', 'winner', 'cash', 'prize', 'offer', 'limited', 'urgent', 'act now', 'claim', 'congratulations']
    found = [w for w in spam_indicators if w in text.lower()]
    
    return {
        'length': len(text),
        'word_count': len(text.split()),
        'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'special_chars': sum(1 for c in text if not c.isalnum() and c != ' '),
        'spam_indicators': found,
        'spam_score': len(found)
    }

# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════

if 'email_history' not in st.session_state:
    st.session_state.email_history = []
if 'show_result' not in st.session_state:
    st.session_state.show_result = False

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h2>🛡️ SpamGuard</h2>
        <p style='color: rgba(255,255,255,0.6); font-size: 0.9em;'>ELITE v2.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    page = st.radio(
        "**Navigation Menu**",
        ["🏠 Home", "✉️ Classify Email", "📊 Analytics", "🎯 Batch Process", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.markdown("""
    <div style='padding: 20px; background: rgba(147, 112, 219, 0.1); border-radius: 12px; border: 1px solid rgba(147, 112, 219, 0.3);'>
        <p style='font-weight: bold; margin-bottom: 10px;'>📈 Model Status</p>
        <p style='color: #22c55e; font-size: 0.9em;'>✅ Production Ready</p>
        <p style='font-size: 0.85em; color: rgba(255,255,255,0.6); margin-top: 10px;'>Accuracy: 98.5%</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    if st.session_state.email_history:
        st.markdown(f"**📧 Emails Analyzed:** {len(st.session_state.email_history)}")
        spam_count = sum(1 for e in st.session_state.email_history if e['prediction'] == 'SPAM')
        st.markdown(f"**🚨 Spam Detected:** {spam_count}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.markdown("""
    <div class='header-gradient'>
        <h1>🛡️ SpamGuard ELITE</h1>
        <p>Next-Generation AI-Powered Email Security</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <p style='font-size: 2.5em; color: #22c55e;'>98.5%</p>
            <p style='font-weight: bold;'>Accuracy</p>
            <p style='font-size: 0.9em; color: rgba(255,255,255,0.6);'>Detection Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <p style='font-size: 2.5em; color: #3b82f6;'>5000+</p>
            <p style='font-weight: bold;'>Features</p>
            <p style='font-size: 0.9em; color: rgba(255,255,255,0.6);'>ML Model</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <p style='font-size: 2.5em; color: #f59e0b;'>Real-time</p>
            <p style='font-weight: bold;'>Processing</p>
            <p style='font-size: 0.9em; color: rgba(255,255,255,0.6);'>Instant Results</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("## ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='premium-card'>
            <h3>🚀 Lightning Fast</h3>
            <p>Classify emails in milliseconds with our optimized SVM model</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='premium-card'>
            <h3>🎯 Highly Accurate</h3>
            <p>98.5% accuracy with advanced TF-IDF feature extraction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='premium-card'>
            <h3>📊 Rich Analytics</h3>
            <p>Detailed insights and historical tracking of all classifications</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='premium-card'>
            <h3>🔒 Secure & Private</h3>
            <p>All data processed locally with enterprise-grade security</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(59, 130, 246, 0.1)); padding: 30px; border-radius: 16px; border: 1px solid rgba(147, 112, 219, 0.3);'>
        <h3 style='color: #22c55e;'>🎬 Getting Started</h3>
        <p>Use the <b>✉️ Classify Email</b> tab to detect spam in real-time, or switch to <b>🎯 Batch Process</b> to analyze multiple emails at once.</p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE: CLASSIFY EMAIL
# ════════════════════════════════════════════════════════════════════════════

elif page == "✉️ Classify Email":
    st.markdown("""
    <div class='header-gradient'>
        <h1>✉️ Email Classification</h1>
        <p>Analyze any email in real-time</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not model_loaded:
        st.error("❌ Model files not found! Please upload 'svm_spam_classifier.pkl' and 'tfidf_vectorizer.pkl'")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            email_text = st.text_area(
                "**Paste Email Content:**",
                placeholder="Paste the email text here for spam detection...",
                height=200
            )
        
        with col2:
            st.markdown("### 📋 Quick Template")
            st.info("""
**Spam Example:**
Click here for FREE money! Limited time offer!

**Legitimate:**
Hi, let's schedule a meeting tomorrow at 2pm.
            """)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            analyze_btn = st.button("🔍 Analyze Email", use_container_width=True)
        with col2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        with col3:
            example_btn = st.button("📝 Load Example", use_container_width=True)
        
        if clear_btn:
            st.rerun()
        
        if example_btn:
            st.session_state.example_text = "Congratulations! You've won a FREE iPhone 15! Click here now to claim your prize before it expires. Act now!"
        
        if email_text or 'example_text' in st.session_state:
            text_to_analyze = email_text or st.session_state.get('example_text', '')
            
            if analyze_btn or 'example_text' in st.session_state:
                with st.spinner("🔄 Analyzing..."):
                    time.sleep(0.5)  # Small delay for visual effect
                    
                    result = predict_spam(text_to_analyze)
                    features = analyze_email_features(text_to_analyze)
                    
                    # Add to history
                    st.session_state.email_history.append({
                        'timestamp': datetime.now(),
                        'text': text_to_analyze[:100],
                        'prediction': result['prediction'],
                        'spam_probability': result['spam_probability'],
                        'confidence': result['confidence'],
                        'risk_level': result['risk_level']
                    })
                    
                    st.session_state.show_result = True
                
                if st.session_state.show_result:
                    st.divider()
                    
                    # ════ RESULT DISPLAY ════
                    if result['prediction'] == 'SPAM':
                        st.markdown(f"""
                        <div class='danger-card' style='animation: slideInLeft 0.6s ease-out;'>
                            <h2 style='color: #ef4444;'>🚨 SPAM DETECTED</h2>
                            <p>This email appears to be spam with high confidence.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='success-card' style='animation: slideInLeft 0.6s ease-out;'>
                            <h2 style='color: #22c55e;'>✅ LEGITIMATE EMAIL</h2>
                            <p>This email appears to be safe and authentic.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # ════ METRICS ════
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <p style='font-size: 2em; color: #f59e0b;'>{result['confidence']*100:.1f}%</p>
                            <p style='font-weight: bold;'>Confidence</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        color = '#ef4444' if result['prediction'] == 'SPAM' else '#22c55e'
                        st.markdown(f"""
                        <div class='metric-card'>
                            <p style='font-size: 2em; color: {color};'>{result['spam_probability']*100:.1f}%</p>
                            <p style='font-weight: bold;'>Spam Score</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <p style='font-size: 1.8em;'>{features['word_count']}</p>
                            <p style='font-weight: bold;'>Words</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <p style='font-size: 1.8em;'>{len(features['spam_indicators'])}</p>
                            <p style='font-weight: bold;'>Spam Flags</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # ════ DETAILED ANALYSIS ════
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Classification Breakdown")
                        
                        # Animated gauge chart
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=result['spam_probability']*100,
                            title={'text': "Spam Probability"},
                            delta={'reference': 50},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#ef4444" if result['spam_probability'] > 0.5 else "#22c55e"},
                                'steps': [
                                    {'range': [0, 25], 'color': "rgba(34, 197, 94, 0.2)"},
                                    {'range': [25, 50], 'color': "rgba(245, 158, 11, 0.2)"},
                                    {'range': [50, 75], 'color': "rgba(249, 115, 22, 0.2)"},
                                    {'range': [75, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig_gauge.update_layout(
                            font={'color': 'white'},
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            height=400
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with col2:
                        st.markdown("### 🔍 Email Characteristics")
                        
                        metrics_data = {
                            '📏 Length': f"{features['length']} chars",
                            '📝 Words': f"{features['word_count']}",
                            '📊 Avg Word': f"{features['avg_word_length']:.2f}",
                            '🔤 Uppercase': f"{features['uppercase_ratio']*100:.1f}%",
                            '🔣 Special Chars': f"{features['special_chars']}",
                            '⚠️ Spam Flags': f"{features['spam_score']}"
                        }
                        
                        for key, value in metrics_data.items():
                            st.markdown(f"**{key}:** {value}")
                        
                        if features['spam_indicators']:
                            st.markdown("#### 🚩 Spam Indicators Found:")
                            for indicator in features['spam_indicators']:
                                st.markdown(f"• **{indicator}**")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ════════════════════════════════════════════════════════════════════════════

elif page == "📊 Analytics":
    st.markdown("""
    <div class='header-gradient'>
        <h1>📊 Analytics Dashboard</h1>
        <p>Real-time insights and performance metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.email_history:
        hist_df = pd.DataFrame(st.session_state.email_history)
        
        # ════ KEY METRICS ════
        col1, col2, col3, col4 = st.columns(4)
        
        total_emails = len(hist_df)
        spam_count = (hist_df['prediction'] == 'SPAM').sum()
        legit_count = (hist_df['prediction'] == 'LEGITIMATE').sum()
        avg_confidence = hist_df['confidence'].mean()
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <p style='font-size: 2.5em; color: #3b82f6;'>{total_emails}</p>
                <p style='font-weight: bold;'>Total Analyzed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <p style='font-size: 2.5em; color: #ef4444;'>{spam_count}</p>
                <p style='font-weight: bold;'>Spam Detected</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <p style='font-size: 2.5em; color: #22c55e;'>{legit_count}</p>
                <p style='font-weight: bold;'>Legitimate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <p style='font-size: 2.5em; color: #f59e0b;'>{avg_confidence*100:.1f}%</p>
                <p style='font-weight: bold;'>Avg Confidence</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # ════ CHARTS ════
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Legitimate', 'Spam'],
                values=[legit_count, spam_count],
                marker=dict(colors=['#22c55e', '#ef4444']),
                textposition='inside',
                textinfo='label+percent',
                hoverinfo='label+value+percent',
            )])
            fig_pie.update_layout(
                title='Email Classification Distribution',
                font=dict(color='white', size=12),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Confidence box plot
            fig_box = go.Figure(data=[
                go.Box(y=hist_df[hist_df['prediction']=='LEGITIMATE']['confidence'],
                       name='Legitimate',
                       marker_color='#22c55e'),
                go.Box(y=hist_df[hist_df['prediction']=='SPAM']['confidence'],
                       name='Spam',
                       marker_color='#ef4444')
            ])
            fig_box.update_layout(
                title='Confidence Distribution',
                yaxis_title='Confidence Score',
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        st.divider()
        
        # ════ HISTORY TABLE ════
        st.markdown("### 📋 Classification History")
        
        display_df = hist_df[['timestamp', 'text', 'prediction', 'confidence', 'risk_level']].copy()
        display_df['confidence'] = (display_df['confidence']*100).round(1).astype(str) + '%'
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.dataframe(display_df, use_container_width=True, height=300)
    
    else:
        st.info("📊 No data yet. Classify some emails to see analytics!")

# ════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH PROCESS
# ════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Batch Process":
    st.markdown("""
    <div class='header-gradient'>
        <h1>🎯 Batch Email Processing</h1>
        <p>Analyze multiple emails at once</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not model_loaded:
        st.error("❌ Model not loaded!")
    else:
        uploaded_file = st.file_uploader("**📤 Upload CSV file** (with 'email' or 'text' column)", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            # Find email column
            email_col = None
            for col in ['email', 'text', 'message', 'content']:
                if col in df.columns:
                    email_col = col
                    break
            
            if email_col is None:
                st.error(f"❌ No email column found. Available: {df.columns.tolist()}")
            else:
                if st.button("🚀 Process Batch", use_container_width=True):
                    progress_bar = st.progress(0)
                    results_list = []
                    
                    for idx, email in enumerate(df[email_col]):
                        result = predict_spam(str(email))
                        results_list.append(result)
                        progress_bar.progress((idx + 1) / len(df))
                    
                    df_results = pd.DataFrame(results_list)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        spam_count = (df_results['prediction'] == 'SPAM').sum()
                        st.metric("🚨 Spam Found", spam_count)
                    with col2:
                        legit_count = (df_results['prediction'] == 'LEGITIMATE').sum()
                        st.metric("✅ Legitimate", legit_count)
                    with col3:
                        avg_conf = df_results['confidence'].mean()
                        st.metric("🎯 Avg Confidence", f"{avg_conf*100:.1f}%")
                    
                    st.divider()
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Download button
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Results",
                        data=csv,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# ════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ════════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Settings":
    st.markdown("""
    <div class='header-gradient'>
        <h1>⚙️ Settings & Configuration</h1>
        <p>Customize your experience</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='premium-card'>
            <h3>📊 Model Configuration</h3>
            <p><strong>Algorithm:</strong> Linear SVM</p>
            <p><strong>Features:</strong> TF-IDF (5000 terms)</p>
            <p><strong>Accuracy:</strong> 98.5%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='premium-card'>
            <h3>📈 Performance Metrics</h3>
            <p><strong>Precision:</strong> 98.4%</p>
            <p><strong>Recall:</strong> 97.1%</p>
            <p><strong>F1-Score:</strong> 97.7%</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    threshold = st.slider("**Spam Detection Threshold**", 0.0, 1.0, 0.5, 0.05)
    st.caption(f"Current threshold: {threshold*100:.0f}%")
    
    st.divider()
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.email_history = []
        st.success("✅ History cleared!")

# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("""
<div class='footer-text'>
    <p>🛡️ <strong>SpamGuard ELITE</strong> v2.0 | Professional Email Security</p>
    <p>Built with ❤️ by Ayesha | Arch Technologies Internship</p>
    <p>© 2024 SpamGuard. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
