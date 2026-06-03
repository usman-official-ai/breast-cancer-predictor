import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

st.markdown("""
<style>
.main { background-color: #f8f9fa; }
.stButton>button { background-color: #2c3e50; color: white; border-radius: 6px; width: 100%; }
.prediction-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; text-align: center; color: white; margin: 1rem 0; }
.footer { text-align: center; margin-top: 3rem; padding: 1rem; font-size: 0.8rem; color: #6c757d; border-top: 1px solid #dee2e6; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    with open('random_forest_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('xgboost_model.pkl', 'rb') as f:
        xgb = pickle.load(f)
    with open('std_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return rf, xgb, scaler

rf_model, xgb_model, scaler = load_models()

feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Model Evaluation", "About"])

if page == "Home":
    st.title("Breast Cancer Prediction App")
    st.markdown("Select a model and adjust the input features to get a prediction.")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Input Features")
        input_values = []
        cols = st.columns(3)
        for i, f in enumerate(feature_names):
            default = 10.0 if "radius" in f or "perimeter" in f else 0.1 if "smoothness" in f else 1.0
            val = cols[i%3].number_input(f, value=default, format="%.4f", key=f)
            input_values.append(val)
    
    with col_right:
        st.subheader("Prediction")
        model_choice = st.selectbox("Select Model", ["Random Forest", "XGBoost"])
        if st.button("Predict", use_container_width=True):
            input_array = np.array(input_values).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            if model_choice == "Random Forest":
                pred = rf_model.predict(input_scaled)[0]
                proba = rf_model.predict_proba(input_scaled)[0]
            else:
                pred = xgb_model.predict(input_scaled)[0]
                proba = xgb_model.predict_proba(input_scaled)[0]
            
            if pred == 1:
                st.markdown(f'<div class="prediction-box"><h2>BENIGN</h2><p>Predicted by {model_choice}</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box"><h2>MALIGNANT</h2><p>Predicted by {model_choice}</p></div>', unsafe_allow_html=True)
            
            prob_df = pd.DataFrame({'Class': ['Malignant', 'Benign'], 'Probability': [proba[0], proba[1]]})
            fig = px.bar(prob_df, x='Class', y='Probability', color='Class',
                         color_discrete_map={'Malignant':'#e74c3c','Benign':'#2ecc71'})
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

elif page == "Model Evaluation":
    st.title("Model Performance Comparison")
    
    data = load_breast_cancer()
    X = data.data
    y = data.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_scaled = scaler.transform(X_test)
    
    rf_pred = rf_model.predict(X_test_scaled)
    rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    st.subheader("Classification Report")
    col1, col2 = st.columns(2)
    with col1:
        st.text("Random Forest")
        st.text(classification_report(y_test, rf_pred, target_names=['Malignant','Benign']))
    with col2:
        st.text("XGBoost")
        st.text(classification_report(y_test, xgb_pred, target_names=['Malignant','Benign']))
    
    st.subheader("Confusion Matrices")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Malignant','Benign'], yticklabels=['Malignant','Benign'], ax=ax[0])
    ax[0].set_title('Random Forest')
    sns.heatmap(confusion_matrix(y_test, xgb_pred), annot=True, fmt='d', cmap='Greens', xticklabels=['Malignant','Benign'], yticklabels=['Malignant','Benign'], ax=ax[1])
    ax[1].set_title('XGBoost')
    st.pyplot(fig)
    
    st.subheader("ROC Curves")
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
    auc_rf = auc(fpr_rf, tpr_rf)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)
    auc_xgb = auc(fpr_xgb, tpr_xgb)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr_rf, tpr_rf, label=f'RF (AUC={auc_rf:.3f})')
    ax_roc.plot(fpr_xgb, tpr_xgb, label=f'XGB (AUC={auc_xgb:.3f})')
    ax_roc.plot([0,1],[0,1],'k--')
    ax_roc.legend()
    st.pyplot(fig_roc)
    
    st.subheader("Histogram of Probabilities")
    fig_hist, ax_hist = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(rf_proba, bins=20, kde=True, ax=ax_hist[0])
    ax_hist[0].axvline(0.5, color='red', linestyle='--')
    ax_hist[0].set_title('Random Forest')
    sns.histplot(xgb_proba, bins=20, kde=True, ax=ax_hist[1])
    ax_hist[1].axvline(0.5, color='red', linestyle='--')
    ax_hist[1].set_title('XGBoost')
    st.pyplot(fig_hist)
    
    st.subheader("Radar Chart - Feature Importances")
    rf_imp = rf_model.feature_importances_
    xgb_imp = xgb_model.feature_importances_
    top_idx = np.argsort(rf_imp)[-8:]
    top_feats = data.feature_names[top_idx]
    rf_top = rf_imp[top_idx]
    xgb_top = xgb_imp[top_idx]
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=rf_top, theta=top_feats, fill='toself', name='RF'))
    fig_radar.add_trace(go.Scatterpolar(r=xgb_top, theta=top_feats, fill='toself', name='XGB'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), title="Top 8 Features")
    st.plotly_chart(fig_radar, use_container_width=True)

else:
    st.title("About")
    st.markdown("""
    **End-to-End Machine Learning Project**
    
    - Dataset: Breast Cancer Wisconsin (Diagnostic)
    - Models: Random Forest and XGBoost
    - Preprocessing: Missing values, normalization, standardization, encoding
    - Split: 80-20 stratified
    - Evaluation: Classification report, confusion matrix, ROC curve, histogram, radar chart
    - Frontend: Streamlit
    - Deployment: Hugging Face Spaces
    
    **Links:**
    - [Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
    - [Colab Notebook](#) (create using the cells above)
    - [Live App](#) (deploy using Hugging Face Spaces)
    """)

st.markdown('<div class="footer">Breast Cancer Predictor | Random Forest + XGBoost | Streamlit</div>', unsafe_allow_html=True)