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

# 3. 界面逻辑
CUTOFF_VALUE = 0.50

st.sidebar.title("📋 Patient Information")
age = st.sidebar.number_input("Age (years)", min_value=18, max_value=100, value=55, step=1)

# 【修改点 1】调整了碳酸氢根的临床合理范围 (单位: mmol/L，参考正常值 ~22-29，SAH重症患者常呈轻度偏低，故默认设为 21.0)
bicarbonate_min = st.sidebar.number_input("Bicarbonate Min (mmol/L)", min_value=10.0, max_value=45.0, value=21.0, step=0.1, format="%.1f")

chloride_min = st.sidebar.number_input("Chloride Min (mmol/L)", min_value=70, max_value=130, value=102, step=1)
treatment_group = st.sidebar.selectbox("Treatment Group", ["Clipping", "Coiling", "EVD Only", "Other"], index=1)

predict_clicked = st.sidebar.button("🚀 Start Prediction", use_container_width=True, type="primary")

st.title("🎯 Cerebral Vasospasm Risk Predictor")
st.markdown("Machine Learning-based Delayed Cerebral Vasospasm Prediction | Optimized & Calibrated Framework")
st.divider()

if predict_clicked:
    # 【修改点 2】将 'icu_los' 替换为 'bicarbonate_min'，确保特征名与新训练模型一致
    input_df = pd.DataFrame({
        'age': [age],
        'bicarbonate_min': [bicarbonate_min],
        'chloride_min': [chloride_min],
        'treatment_group': [treatment_group]
    })
    
    st.subheader("📋 Input Data Preview")
    st.dataframe(input_df, use_container_width=True)
    
    # 推理逻辑
    if final_model is not None:
        try:
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
            
            # 建议模块
            if is_high_risk:
                st.error("🚨 IMMEDIATE ACTION REQUIRED: Initiate Nimodipine, perform daily TCD...")
            else:
                st.success("✅ CONTINUED MONITORING: Weekly liver function tests, outpatient follow-up...")
        except Exception as e:
            st.error(f"⚠️ 预测计算出错，请检查输入特征列名是否与模型训练时完全一致。错误信息: {e}")
    else:
        st.warning("模型未加载，无法进行预测。")