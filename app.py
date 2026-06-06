"""
Breast Cancer Prediction System
Author: Usman Official
Model: Random Forest with Pipeline
Deployment: Streamlit Cloud
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

# Page Configuration
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f0f2f6;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e3c72;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Prediction box */
    .benign-box {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .malignant-box {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1e3c72;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1.5rem;
        background-color: #1e3c72;
        color: white;
        border-radius: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1e3c72;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #2ecc71;
        color: white;
        transform: translateY(-2px);
    }
    
    /* Metrics */
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load Pipeline
@st.cache_resource
def load_production_pipeline():
    with open('breast_cancer_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

try:
    pipeline = load_production_pipeline()
    model_loaded = True
except FileNotFoundError:
    st.error("❌ Model pipeline not found. Please ensure 'breast_cancer_pipeline.pkl' is in the directory.")
    model_loaded = False

# Load feature names
data = load_breast_cancer()
feature_names = data.feature_names.tolist()
target_names = ['Malignant', 'Benign']

# Sidebar Navigation
st.sidebar.title("🎗️ Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📊 Model Evaluation", "ℹ️ About"],
    format_func=lambda x: x
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Dataset:** Breast Cancer Wisconsin\n"
    "**Model:** Random Forest\n"
    "**Accuracy:** 97%+"
)

# Home Page
if page == "🏠 Home":
    st.title("🎗️ Breast Cancer Prediction System")
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
        <p style='font-size: 1.1rem; color: #1e3c72;'>
        This system uses a <strong>Random Forest Classifier</strong> to predict whether a breast tumor is 
        <strong>Malignant</strong> (cancerous) or <strong>Benign</strong> (non-cancerous) based on 30 cellular measurements.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📏 Input Features")
        st.markdown("Adjust the values below to get a real-time prediction.")
        
        input_values = []
        cols = st.columns(3)
        for i, feature in enumerate(feature_names):
            default_value = 12.0 if "radius" in feature or "perimeter" in feature else \
                           0.1 if "smoothness" in feature or "concavity" in feature else \
                           1.0 if "error" in feature else \
                           100.0 if "area" in feature else 0.5
            val = cols[i%3].number_input(
                feature,
                value=default_value,
                format="%.4f",
                key=f"input_{i}",
                help=f"Enter the {feature} value"
            )
            input_values.append(val)
    
    with col2:
        st.subheader("🎯 Prediction Result")
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔍 Predict", use_container_width=True):
            if model_loaded:
                input_array = np.array(input_values).reshape(1, -1)
                prediction = pipeline.predict(input_array)[0]
                probabilities = pipeline.predict_proba(input_array)[0]
                
                if prediction == 1:
                    st.markdown("""
                    <div class="benign-box">
                        <h1>✅ BENIGN</h1>
                        <p style="font-size: 1.2rem;">The tumor appears to be <strong>non-cancerous</strong>.</p>
                        <p style="font-size: 1rem;">Recommended: Regular monitoring and follow-up.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="malignant-box">
                        <h1>⚠️ MALIGNANT</h1>
                        <p style="font-size: 1.2rem;">The tumor shows <strong>cancerous</strong> characteristics.</p>
                        <p style="font-size: 1rem;">Recommended: Immediate consultation with an oncologist.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability Chart
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
                    title="Prediction Confidence"
                )
                fig.update_layout(
                    showlegend=False,
                    height=350,
                    plot_bgcolor='white',
                    title_font_size=16
                )
                fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence meter
                confidence = max(probabilities)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Model Confidence</h4>
                    <div style="font-size: 2rem; font-weight: bold; color: #1e3c72;">{confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Model not loaded. Please check the pipeline file.")

# Model Evaluation Page
elif page == "📊 Model Evaluation":
    st.title("📊 Model Performance Analysis")
    st.markdown("---")
    
    if model_loaded:
        # Load test data
        X = data.data
        y = data.target
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{pipeline.score(X_test, y_test):.2%}")
        with col2:
            from sklearn.metrics import precision_score
            st.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
        with col3:
            from sklearn.metrics import recall_score
            st.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
        with col4:
            from sklearn.metrics import f1_score
            st.metric("F1-Score", f"{f1_score(y_test, y_pred):.2%}")
        
        st.markdown("---")
        
        # Classification Report
        with st.expander("📋 Classification Report", expanded=True):
            report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.3f}").background_gradient(cmap="Blues"))
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("🔢 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
            ax_cm.set_ylabel('Actual')
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_title('Confusion Matrix')
            st.pyplot(fig_cm)
        
        with col_right:
            st.subheader("📈 ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.fill_between(fpr, tpr, alpha=0.3, color='orange')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('ROC Curve')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
        
        st.subheader("📊 Histogram of Predicted Probabilities")
        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        sns.histplot(y_proba, bins=25, kde=True, color='steelblue', edgecolor='black')
        ax_hist.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
        ax_hist.set_xlabel('Probability of Benign')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Distribution of Model Confidence')
        ax_hist.legend()
        st.pyplot(fig_hist)
        
        st.subheader("📡 Feature Importance Radar Chart")
        importances = pipeline.named_steps['classifier'].feature_importances_
        top_indices = np.argsort(importances)[-8:]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=top_importances,
            theta=top_features,
            fill='toself',
            name='Feature Importance',
            line_color='#1e3c72',
            fillcolor='rgba(30, 60, 114, 0.3)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(top_importances) * 1.1])
            ),
            title="Top 8 Most Important Features",
            showlegend=True,
            height=600
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# About Page
else:
    st.title("ℹ️ About This Project")
    st.markdown("---")
    
    col_info, col_links = st.columns([2, 1])
    
    with col_info:
        st.markdown("""
        ### 🎯 Project Overview
        
        This is a **complete end-to-end machine learning project** that demonstrates:
        
        | Component | Description |
        |-----------|-------------|
        | **Dataset** | Breast Cancer Wisconsin (Diagnostic) - 569 samples, 30 features |
        | **Preprocessing** | Missing value handling, Normalization (MinMaxScaler), Standardization (StandardScaler) |
        | **Model** | Random Forest Classifier with 150 estimators |
        | **Pipeline** | StandardScaler + RandomForestClassifier |
        | **Split Ratio** | 80% Training, 20% Testing (Stratified) |
        | **Evaluation Metrics** | Classification Report, Confusion Matrix, ROC Curve, Histogram, Radar Chart |
        | **Frontend** | Streamlit Web Application |
        | **Deployment** | Streamlit Community Cloud |
        
        ### 📊 Model Performance
        
        - **Accuracy:** >97%
        - **AUC Score:** >0.99
        - **Precision (Benign):** 0.98
        - **Recall (Benign):** 0.97
        
        ### 🔧 Technologies Used
        
        - Python 3.9+
        - scikit-learn (Random Forest)
        - pandas, numpy (Data Processing)
        - matplotlib, seaborn, plotly (Visualization)
        - Streamlit (Frontend)
        - Git & GitHub (Version Control)
        """)
    
    with col_links:
        st.markdown("""
        ### 🔗 Important Links
        
        - [📓 Google Colab Notebook](#)
        - [📊 Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
        - [💻 GitHub Repository](https://github.com/usman-official-ai/breast-cancer-predictor)
        - [🌐 Live Application](#)
        
        ### 👨‍💻 Author
        
        **Usman Official**
        
        - GitHub: [usman-official-ai](https://github.com/usman-official-ai)
        
        ### 📄 License
        
        MIT License - Free for academic and commercial use.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🎗️ Breast Cancer Prediction System | Powered by Random Forest & Streamlit | © 2024</p>
    <p style="font-size: 0.8rem; opacity: 0.8;">For educational purposes. Always consult a medical professional for diagnosis.</p>
</div>
""", unsafe_allow_html=True)