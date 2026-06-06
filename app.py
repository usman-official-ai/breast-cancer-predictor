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

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
.stButton>button { background-color: #2c3e50; color: white; border-radius: 6px; width: 100%; }
.prediction-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; text-align: center; color: white; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    with open('breast_cancer_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

pipeline = load_pipeline()
feature_names = load_breast_cancer().feature_names.tolist()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Model Evaluation", "About"])

if page == "Home":
    st.title("Breast Cancer Prediction")
    input_values = []
    cols = st.columns(3)
    for i, f in enumerate(feature_names):
        default = 10.0 if "radius" in f else 0.1
        val = cols[i%3].number_input(f, value=default, format="%.4f", key=f)
        input_values.append(val)

    if st.button("Predict"):
        arr = np.array(input_values).reshape(1, -1)
        pred = pipeline.predict(arr)[0]
        proba = pipeline.predict_proba(arr)[0][1]
        if pred == 1:
            st.markdown('<div class="prediction-box"><h2>BENIGN</h2></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-box"><h2>MALIGNANT</h2></div>', unsafe_allow_html=True)
        st.write(f"Probability benign: {proba:.3f}")

elif page == "Model Evaluation":
    st.title("Model Performance")
    data = load_breast_cancer()
    X, y = data.data, data.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:,1]

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=['Malignant','Benign']))

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = auc(fpr, tpr)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f'AUC={auc_score:.3f}')
    ax2.plot([0,1],[0,1],'k--')
    ax2.legend()
    st.pyplot(fig2)

else:
    st.title("About")
    st.markdown("Model: Random Forest | Dataset: Wisconsin Breast Cancer")