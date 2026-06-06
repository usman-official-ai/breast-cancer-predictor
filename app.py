"""
Breast Cancer Prediction System
================================
Production-grade ML application for breast cancer diagnosis
Model: Random Forest with Pipeline
Deployed on: Streamlit Cloud
Author: Usman Official
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Breast Cancer Predictor | AI Healthcare",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR PROFESSIONAL LOOK
# ============================================================================
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f5f7fb;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a3c5e;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    
    h1 {
        border-bottom: 3px solid #2ecc71;
        display: inline-block;
        padding-bottom: 10px;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    
    /* Prediction boxes */
    .benign-box {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px rgba(46,204,113,0.3);
        animation: fadeIn 0.5s ease-in;
    }
    
    .malignant-box {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px rgba(231,76,60,0.3);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a3c5e 0%, #0d2b45 100%);
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 4px solid #2ecc71;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1a3c5e;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #1a3c5e 0%, #0d2b45 100%);
        color: white;
        border-radius: 30px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(46,204,113,0.3);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        background: #1a3c5e;
        color: white;
        border-radius: 12px;
        font-size: 0.8rem;
    }
    
    .footer a {
        color: #2ecc71;
        text-decoration: none;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD PIPELINE
# ============================================================================
@st.cache_resource
def load_production_pipeline():
    try:
        with open('breast_cancer_pipeline.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Model pipeline not found. Please contact administrator.")
        return None

pipeline = load_production_pipeline()

if pipeline is None:
    st.stop()

# Load data for feature names
data = load_breast_cancer()
feature_names = data.feature_names.tolist()
target_names = ['Malignant (Cancerous)', 'Benign (Non-cancerous)']

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("🎗️ Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page",
    ["🏠 Prediction Dashboard", "📊 Model Performance", "ℹ️ About Project"],
    format_func=lambda x: x
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**📊 Dataset:** Breast Cancer Wisconsin\n\n"
    "**🤖 Model:** Random Forest\n\n"
    "**🎯 Accuracy:** 97%+\n\n"
    "**📈 AUC:** 0.99"
)

st.sidebar.markdown("---")
st.sidebar.caption("© 2024 | AI Healthcare System")

# ============================================================================
# PAGE 1: PREDICTION DASHBOARD
# ============================================================================
if page == "🏠 Prediction Dashboard":
    st.title("🎗️ Breast Cancer Prediction System")
    
    st.markdown("""
    <div style='background: #e8f4f8; padding: 1.2rem; border-radius: 12px; margin-bottom: 2rem; border-left: 4px solid #2ecc71;'>
        <p style='font-size: 1rem; color: #1a3c5e; margin: 0;'>
        <strong>Clinical Decision Support System</strong> – Enter 30 cellular measurements to classify breast tumors as 
        <strong style='color: #2ecc71;'>Benign (Non-cancerous)</strong> or 
        <strong style='color: #e74c3c;'>Malignant (Cancerous)</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_result = st.columns([2, 1])
    
    with col_input:
        st.subheader("📏 Patient Measurements")
        st.markdown("*Adjust the values below using the sliders or number inputs.*")
        
        input_values = []
        cols = st.columns(3)
        for i, feature in enumerate(feature_names):
            default_value = 12.0 if "radius" in feature else \
                           0.1 if "smoothness" in feature else \
                           1.0 if "error" in feature else \
                           100.0 if "area" in feature else 0.5
            val = cols[i%3].number_input(
                feature,
                value=default_value,
                format="%.4f",
                key=f"feat_{i}",
                help=f"Enter the {feature} value"
            )
            input_values.append(val)
    
    with col_result:
        st.subheader("🎯 Diagnosis Result")
        
        if st.button("🔍 Run Prediction", use_container_width=True):
            input_array = np.array(input_values).reshape(1, -1)
            prediction = pipeline.predict(input_array)[0]
            probabilities = pipeline.predict_proba(input_array)[0]
            
            if prediction == 1:
                st.markdown("""
                <div class="benign-box">
                    <h1 style="font-size: 2.5rem; margin: 0;">✅ BENIGN</h1>
                    <p style="font-size: 1rem; margin-top: 0.5rem;">Non-cancerous tumor</p>
                    <hr style="margin: 1rem 0; border-color: rgba(255,255,255,0.2);">
                    <p style="margin: 0;">Recommended: Regular monitoring and follow-up screening.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="malignant-box">
                    <h1 style="font-size: 2.5rem; margin: 0;">⚠️ MALIGNANT</h1>
                    <p style="font-size: 1rem; margin-top: 0.5rem;">Cancerous tumor detected</p>
                    <hr style="margin: 1rem 0; border-color: rgba(255,255,255,0.2);">
                    <p style="margin: 0;">Recommended: Immediate consultation with an oncologist.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability Bar Chart
            prob_df = pd.DataFrame({
                'Class': ['Malignant', 'Benign'],
                'Probability': [probabilities[0], probabilities[1]]
            })
            
            fig = px.bar(
                prob_df,
                x='Class',
                y='Probability',
                color='Class',
                color_discrete_map={'Malignant': '#e74c3c', 'Benign': '#2ecc71'},
                title="Prediction Confidence",
                text=prob_df['Probability'].apply(lambda x: f'{x:.1%}')
            )
            fig.update_layout(
                showlegend=False,
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                title_font_size=16
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence score
            confidence = max(probabilities)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{confidence:.1%}</div>
                <div class="metric-label">Model Confidence Score</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: MODEL PERFORMANCE
# ============================================================================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Analysis")
    st.markdown("*Comprehensive evaluation metrics for the Random Forest classifier*")
    st.markdown("---")
    
    # Prepare test data
    X = data.data
    y = data.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Key Metrics Row
    st.subheader("📈 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        acc = pipeline.score(X_test, y_test)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{acc:.1%}</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        from sklearn.metrics import precision_score
        prec = precision_score(y_test, y_pred)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{prec:.1%}</div>
            <div class="metric-label">Precision</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        from sklearn.metrics import recall_score
        rec = recall_score(y_test, y_pred)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{rec:.1%}</div>
            <div class="metric-label">Recall</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, y_pred)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{f1:.1%}</div>
            <div class="metric-label">F1 Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{roc_auc:.3f}</div>
            <div class="metric-label">AUC Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Classification Report", "🔢 Confusion Matrix", "📈 ROC Curve", 
        "📊 Histogram Analysis", "📡 Feature Importance"
    ])
    
    with tab1:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.3f}").background_gradient(cmap="Blues"))
    
    with tab2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Malignant', 'Benign'],
                   yticklabels=['Malignant', 'Benign'],
                   ax=ax_cm, cbar=False)
        ax_cm.set_ylabel('Actual Label', fontsize=12)
        ax_cm.set_xlabel('Predicted Label', fontsize=12)
        ax_cm.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        st.pyplot(fig_cm)
    
    with tab3:
        st.subheader("ROC Curve")
        fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
        ax_roc.plot(fpr, tpr, color='#2ecc71', lw=2, label=f'Random Forest (AUC = {roc_auc:.3f})')
        ax_roc.plot([0, 1], [0, 1], color='#e74c3c', lw=2, linestyle='--', label='Random Classifier')
        ax_roc.fill_between(fpr, tpr, alpha=0.2, color='#2ecc71')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        ax_roc.legend(loc='lower right')
        ax_roc.grid(alpha=0.3)
        st.pyplot(fig_roc)
    
    with tab4:
        st.subheader("Histogram of Predicted Probabilities")
        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        sns.histplot(y_proba, bins=25, kde=True, color='#1a3c5e', edgecolor='black')
        ax_hist.axvline(x=0.5, color='#e74c3c', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
        ax_hist.set_xlabel('Probability of Benign', fontsize=12)
        ax_hist.set_ylabel('Frequency', fontsize=12)
        ax_hist.set_title('Distribution of Model Predictions', fontsize=14)
        ax_hist.legend()
        st.pyplot(fig_hist)
    
    with tab5:
        st.subheader("Radar Chart – Feature Importances (Top 8)")
        importances = pipeline.named_steps['rf'].feature_importances_
        top_idx = np.argsort(importances)[-8:]
        top_features = [feature_names[i] for i in top_idx]
        top_importances = importances[top_idx]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=top_importances,
            theta=top_features,
            fill='toself',
            name='Feature Importance',
            line_color='#1a3c5e',
            fillcolor='rgba(26, 60, 94, 0.3)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(top_importances) * 1.1])
            ),
            title="Top 8 Most Important Features",
            showlegend=True,
            height=600,
            title_font_size=16
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ============================================================================
# PAGE 3: ABOUT
# ============================================================================
else:
    st.title("ℹ️ About the System")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        
        This is a **production-grade machine learning system** developed for early breast cancer detection using cellular measurements.
        
        | Component | Specification |
        |-----------|---------------|
        | **Dataset** | Breast Cancer Wisconsin (Diagnostic) – 569 samples, 30 features |
        | **Preprocessing** | StandardScaler for feature normalization |
        | **Model** | Random Forest Classifier (150 estimators, max_depth=10) |
        | **Pipeline** | StandardScaler → RandomForestClassifier |
        | **Train/Test Split** | 80% training, 20% testing (stratified) |
        | **Evaluation Metrics** | Accuracy, Precision, Recall, F1 Score, AUC-ROC |
        | **Visualizations** | Classification Report, Confusion Matrix, ROC Curve, Histogram, Radar Chart |
        | **Frontend** | Streamlit Web Application |
        | **Deployment** | Streamlit Cloud |
        
        ### 📊 Model Performance Summary
        
        - **Accuracy:** 97%+
        - **AUC Score:** 0.99
        - **Precision (Benign):** 0.98
        - **Recall (Benign):** 0.97
        - **F1 Score:** 0.97
        
        ### 🔧 Technology Stack
        
        - **Python 3.9+** – Core programming language
        - **scikit-learn** – Machine learning library (Random Forest)
        - **pandas / numpy** – Data processing and manipulation
        - **matplotlib / seaborn / plotly** – Data visualization
        - **Streamlit** – Interactive web frontend
        - **Git & GitHub** – Version control and deployment
        """)
    
    with col2:
        st.markdown("""
        ### 🔗 Quick Links
        
        - [📓 Colab Notebook](#)
        - [📊 Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
        - [💻 GitHub Repository](https://github.com/usman-official-ai/breast-cancer-predictor)
        - [🌐 Live Application](#)
        
        ### 👨‍💻 Developer
        
        **Usman Official**
        
        *AI/ML Engineer*
        
        [GitHub Profile](https://github.com/usman-official-ai)
        
        ### 📄 License
        
        MIT License – Free for academic and healthcare research.
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='background: #1a3c5e; padding: 1rem; border-radius: 10px; text-align: center; color: white;'>
        <p style='margin: 0;'>🎗️ This tool is for educational and research purposes only. Always consult a medical professional for clinical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    <p>🎗️ Breast Cancer Prediction System | Powered by Random Forest & Streamlit</p>
    <p style="font-size: 0.7rem; opacity: 0.8;">Clinical Decision Support System – Research Use Only</p>
</div>
""", unsafe_allow_html=True)