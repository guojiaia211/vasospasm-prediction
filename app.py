import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. 页面基础配置 (必须放在脚本最开始)
st.set_page_config(
    page_title="Cerebral Vasospasm Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 加载训练好的模型
@st.cache_resource # 缓存模型，避免每次点按钮都重新加载
def load_model():
    # 替换为你实际保存的模型路径
    try:
        return joblib.load("best_vasospasm_model.pkl")
    except Exception as e:
        st.warning("⚠️ 模型文件未找到，当前使用[模拟演示模式]。请确保 'best_vasospasm_model.pkl' 存在。")
        return None

final_model = load_model()

# 临床决策阈值
CUTOFF_VALUE = 0.50

# 3. 侧边栏：临床信息输入
st.sidebar.title("📋 Patient Information")
st.sidebar.markdown("Please enter clinical and surgical data.")

age = st.sidebar.number_input("Age (years)", min_value=18, max_value=100, value=55, step=1)
icu_los = st.sidebar.number_input("ICU LOS (days)", min_value=1, max_value=100, value=5, step=1)
chloride_min = st.sidebar.number_input("Chloride Min (mmol/L)", min_value=70, max_value=130, value=102, step=1)
treatment_group = st.sidebar.selectbox("Treatment Group", ["Clipping", "Coiling", "EVD Only", "Other"], index=1)

# 预测按钮
predict_clicked = st.sidebar.button("🚀 Start Prediction", use_container_width=True, type="primary")

# 4. 主页面：标题与说明
st.title("🎯 Cerebral Vasospasm Risk Predictor")
st.markdown("<p style='color: gray; font-size: 16px;'>Machine Learning-based Delayed Cerebral Vasospasm Prediction | Optimized & Calibrated Framework</p>", unsafe_allow_html=True)
st.divider()

# 5. 后端推理与前端渲染逻辑
if predict_clicked:
    # 组装输入数据框 (列名必须与训练时 preprocessor 的要求严格一致)
    input_df = pd.DataFrame({
        'age': [age],
        'icu_los': [icu_los],
        'chloride_min': [chloride_min],
        'treatment_group': [treatment_group]
    })
    
    st.subheader("📋 Input Data Preview")
    st.dataframe(input_df, use_container_width=True)
    
    # 获取预测概率
    if final_model is not None:
        risk_prob = final_model.predict_proba(input_df)[0][1]
    else:
        # 模拟计算逻辑 (仅供演示)
        base_score = -2 + (icu_los * 0.05) - (age * 0.01) - (chloride_min * 0.02)
        if treatment_group == "Coiling": base_score -= 0.5
        risk_prob = 1 / (1 + np.exp(-base_score))
        
    is_high_risk = risk_prob >= CUTOFF_VALUE
    
    st.markdown("<br><h4>📊 Assessment Results</h4>", unsafe_allow_html=True)
    
    # 结果看板展示 (使用 HTML/CSS 还原出 SCI 质感的渐变卡片)
    col1, col2 = st.columns(2)
    
    with col1:
        if is_high_risk:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff6b6b, #ee5a52); color: white; padding: 30px; border-radius: 15px; text-align: center; min-height: 140px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="margin: 0; font-weight: bold; color: white;">🚨 HIGH RISK</h2>
                <h5 style="margin: 10px 0; opacity: 0.95; color: white;">Vasospasm Classification</h5>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 30px; border-radius: 15px; text-align: center; min-height: 140px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="margin: 0; font-weight: bold; color: white;">✅ LOW RISK</h2>
                <h5 style="margin: 10px 0; opacity: 0.95; color: white;">Vasospasm Classification</h5>
            </div>
            """, unsafe_allow_html=True)
            
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4ecdc4, #44a08d); color: white; padding: 30px; border-radius: 15px; text-align: center; min-height: 140px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="margin: 0; font-weight: bold; font-size: 2.5em; color: white;">{risk_prob * 100:.1f}%</h1>
            <h5 style="margin: 10px 0; opacity: 0.95; color: white;">Probability of Vasospasm (Cutoff: {CUTOFF_VALUE*100}%)</h5>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><h4>💉 Clinical Management Recommendations</h4>", unsafe_allow_html=True)
    
    # 临床建议卡片
    if is_high_risk:
        st.error(f"""
        **🚨 IMMEDIATE ACTION REQUIRED (Probability ≥ {CUTOFF_VALUE*100}%):**
        - Initiate or optimize **Nimodipine** therapy strictly.
        - Perform daily Transcranial Doppler (TCD) ultrasonography.
        - Maintain strict euvolemia and consider induced hypertension if symptomatic.
        - **Prepare for potential endovascular rescue therapy.**
        """)
    else:
        st.success(f"""
        **✅ CONTINUED MONITORING (Probability < {CUTOFF_VALUE*100}%):**
        - Standard neuro-ICU observation and routine care.
        - Routine neurologic exams every 4 hours.
        - Continue baseline fluid management.
        - Alert physician immediately if new focal neurological deficits occur.
        """)

else:
    # 初始状态下的占位符
    st.info("👈 Please enter patient information in the sidebar and click **Start Prediction**.")

# 6. 页脚
st.markdown("""
---
<div style="text-align: center; color: gray; font-size: 14px;">
    <p>© 2026 Cerebral Vasospasm Prediction Model | External Validated via eICU Cohort</p>
    <p>Model Architecture: Optimized ML Pipeline with Platt Scaling Calibration</p>
</div>
""", unsafe_allow_html=True)