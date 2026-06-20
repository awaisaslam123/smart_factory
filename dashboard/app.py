"""
AI Smart Factory Intelligence Platform — Main Dashboard
=======================================================
Upgraded with:
  - LSTM Predictive Maintenance
  - Hugging Face NLP Report Generator
  - 4 pages: Historical | Risk Radar | Maintenance | AI Reports | Model Monitor

Run: streamlit run dashboard/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="AI Smart Factory Intelligence Platform",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏭 AI Smart Factory Intelligence Platform")
st.markdown("""
Welcome to the Smart Factory ecosystem. Please select a module from the sidebar to begin:

- **📊 Logistics Spend**: Analyze raw material shipping costs and regional spend.
- **🎯 Risk Radar**: XGBoost-powered predictive delay detection for logistics.
- **🔮 Predictive Maintenance**: LSTM machine failure forecasting for looms and motors.
- **🤖 AI Reports**: HuggingFace NLP executive reports.
- **⚙️ Model Monitor**: Accuracy metrics and drift detection.
""")

with st.sidebar:
    st.success("Select a page above to navigate.")
