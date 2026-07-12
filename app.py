import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. 页面基础配置
st.set_page_config(
    page_title="Cerebral Vasospasm Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 优化模型加载逻辑 (适配 Streamlit Cloud 的文件系统)
@st.cache_resource
def load_model():
    # 查找模型文件的标准做法
    model_filename = "best_vasospasm_model.pkl"
    if os.path.exists(model_filename):
        return joblib.load(model_filename)
    else:
        # 如果当前目录下找不到，尝试在根目录找
        st.error(f"⚠️ 模型文件 {model_filename} 未找到。请确保它与 app.py 在同一个 GitHub 仓库的根目录下。")
        return None

# 加载模型
final_model = load_model()

# 3. 界面逻辑 (保持你的代码逻辑不变，已验证良好)
CUTOFF_VALUE = 0.50

st.sidebar.title("📋 Patient Information")
age = st.sidebar.number_input("Age (years)", min_value=18, max_value=100, value=55, step=1)
icu_los = st.sidebar.number_input("ICU LOS (days)", min_value=1, max_value=100, value=5, step=1)
chloride_min = st.sidebar.number_input("Chloride Min (mmol/L)", min_value=70, max_value=130, value=102, step=1)
treatment_group = st.sidebar.selectbox("Treatment Group", ["Clipping", "Coiling", "EVD Only", "Other"], index=1)

predict_clicked = st.sidebar.button("🚀 Start Prediction", use_container_width=True, type="primary")

st.title("🎯 Cerebral Vasospasm Risk Predictor")
st.markdown("Machine Learning-based Delayed Cerebral Vasospasm Prediction | Optimized & Calibrated Framework")
st.divider()

if predict_clicked:
    input_df = pd.DataFrame({
        'age': [age],
        'icu_los': [icu_los],
        'chloride_min': [chloride_min],
        'treatment_group': [treatment_group]
    })
    
    st.subheader("📋 Input Data Preview")
    st.dataframe(input_df, use_container_width=True)
    
    # 推理逻辑
    if final_model is not None:
        risk_prob = final_model.predict_proba(input_df)[0][1]
        is_high_risk = risk_prob >= CUTOFF_VALUE
        
        col1, col2 = st.columns(2)
        with col1:
            if is_high_risk:
                st.error("🚨 HIGH RISK (Vasospasm)")
            else:
                st.success("✅ LOW RISK (Non-Vasospasm)")
        with col2:
            st.metric("Probability of Vasospasm", f"{risk_prob * 100:.1f}%")
        
        # 建议模块 (保持你的逻辑)
        if is_high_risk:
            st.error("🚨 IMMEDIATE ACTION REQUIRED: Initiate Nimodipine, perform daily TCD...")
        else:
            st.success("✅ CONTINUED MONITORING: Weekly liver function tests, outpatient follow-up...")
    else:
        st.warning("模型未加载，无法进行预测。")