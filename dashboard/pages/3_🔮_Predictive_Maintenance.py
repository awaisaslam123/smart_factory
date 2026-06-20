"""
Page 2 — Predictive Maintenance (LSTM)
Real-time machine health scores, failure forecasts, RUL countdown.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background: #0d1117; color: #e6edf3; }
.metric-card {
    background: linear-gradient(135deg, #161b22, #21262d);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 1.2rem; text-align: center; margin-bottom: 1rem;
}
.metric-value { font-size: 2.2rem; font-weight: 800; }
.sev-critical { color: #ff4d4f; }
.sev-high     { color: #ff7a00; }
.sev-medium   { color: #fadb14; }
.sev-low      { color: #52c41a; }
</style>
""", unsafe_allow_html=True)

BASE = os.path.join(os.path.dirname(__file__), "..", "..")
HEALTH_CSV = os.path.join(BASE, "data", "processed", "machine_health.csv")
SENSOR_CSV = os.path.join(BASE, "data", "sensor", "sensor_data.csv")

# ── Load or generate health data ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_health():
    if os.path.exists(HEALTH_CSV):
        return pd.read_csv(HEALTH_CSV)
    # Generate mock data if not yet trained
    np.random.seed(42)
    machines = [f"M-{i:03d}" for i in range(1, 11)]
    probs = np.random.beta(2, 5, 10)
    return pd.DataFrame({
        "machine_id":          machines,
        "failure_probability": probs.round(4),
        "health_score":        ((1 - probs) * 100).round(1),
        "rul_hours":           (((1 - probs) * 72).astype(int)),
        "severity":            pd.cut(probs, bins=[0,.25,.5,.75,1],
                                      labels=["LOW","MEDIUM","HIGH","CRITICAL"]),
        "last_temperature":    np.random.uniform(55,95,10).round(1),
        "last_vibration":      np.random.uniform(0.3,2.5,10).round(4),
        "last_pressure":       np.random.uniform(70,140,10).round(1),
        "last_rpm":            np.random.uniform(900,1800,10).round(1),
        "last_current":        np.random.uniform(8,20,10).round(3),
    })

@st.cache_data(ttl=600)
def load_sensor_history():
    if os.path.exists(SENSOR_CSV):
        df = pd.read_csv(SENSOR_CSV, parse_dates=["timestamp"])
        return df
    return pd.DataFrame()

df = load_health()
sensor_df = load_sensor_history()

def localize_machine_name(m_id):
    mapping = {
        "M-001": "Lahore Loom Unit-1",
        "M-002": "Lahore Loom Unit-2",
        "M-003": "Faisalabad Spinning Motor-A",
        "M-004": "Faisalabad Spinning Motor-B",
        "M-005": "Multan Weaving Machine-1",
        "M-006": "Multan Weaving Machine-2",
        "M-007": "Karachi Dyeing Vat-A",
        "M-008": "Karachi Dyeing Vat-B",
        "M-009": "Rawalpindi Boiler-1",
        "M-010": "Sialkot Compressor-1"
    }
    return mapping.get(m_id, m_id)

df["machine_id"] = df["machine_id"].apply(localize_machine_name)

st.markdown("# 🔮 Predictive Maintenance Dashboard")
st.markdown("**LSTM-powered failure forecasting** · Real-time equipment health monitoring")
st.divider()

# ── KPI Row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
critical_count = len(df[df["severity"] == "CRITICAL"])
high_count     = len(df[df["severity"] == "HIGH"])
avg_health     = df["health_score"].mean()
min_rul        = df["rul_hours"].min()

with c1:
    st.markdown(f"""<div class="metric-card">
        <div style="color:#aaa;font-size:.85rem">CRITICAL MACHINES</div>
        <div class="metric-value sev-critical">{critical_count}</div>
        <div style="color:#aaa;font-size:.8rem">Immediate action needed</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div style="color:#aaa;font-size:.85rem">HIGH RISK</div>
        <div class="metric-value sev-high">{high_count}</div>
        <div style="color:#aaa;font-size:.8rem">Schedule within 24h</div>
    </div>""", unsafe_allow_html=True)
with c3:
    color = "#52c41a" if avg_health > 70 else "#fadb14" if avg_health > 50 else "#ff4d4f"
    st.markdown(f"""<div class="metric-card">
        <div style="color:#aaa;font-size:.85rem">AVG FLEET HEALTH</div>
        <div class="metric-value" style="color:{color}">{avg_health:.1f}%</div>
        <div style="color:#aaa;font-size:.8rem">Across all machines</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div style="color:#aaa;font-size:.85rem">MIN RUL</div>
        <div class="metric-value sev-critical">{min_rul}h</div>
        <div style="color:#aaa;font-size:.8rem">Shortest remaining life</div>
    </div>""", unsafe_allow_html=True)

st.markdown("### 🏭 Machine Health Heatmap")

# ── Health Heatmap ────────────────────────────────────────────────────────────
color_map = {"CRITICAL": "#ff4d4f", "HIGH": "#ff7a00", "MEDIUM": "#fadb14", "LOW": "#52c41a"}

fig_health = go.Figure()
for _, row in df.iterrows():
    fig_health.add_trace(go.Bar(
        x=[row["machine_id"]], y=[row["health_score"]],
        marker_color=color_map.get(str(row["severity"]), "#58a6ff"),
        name=str(row["severity"]),
        hovertemplate=(
            f"<b>{row['machine_id']}</b><br>"
            f"Health: {row['health_score']:.1f}%<br>"
            f"Failure Prob: {row['failure_probability']:.1%}<br>"
            f"RUL: {row['rul_hours']}h<br>"
            f"Severity: {row['severity']}<extra></extra>"
        )
    ))
fig_health.update_layout(
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
    font=dict(color="#e6edf3"), showlegend=False,
    yaxis=dict(title="Health Score (%)", range=[0, 100], gridcolor="#21262d"),
    xaxis=dict(title="Machine ID", gridcolor="#21262d"),
    height=350, margin=dict(l=40, r=20, t=20, b=40)
)
st.plotly_chart(fig_health, use_container_width=True)

# ── Failure Probability Gauges ─────────────────────────────────────────────────
st.markdown("### 🎯 Failure Probability Gauges (Top 4 at Risk)")
top4 = df.head(4)
cols = st.columns(4)
for i, (_, row) in enumerate(top4.iterrows()):
    prob = row["failure_probability"]
    gauge_color = color_map.get(str(row["severity"]), "#58a6ff")
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 22, "color": gauge_color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#aaa"},
            "bar": {"color": gauge_color},
            "bgcolor": "#21262d",
            "steps": [
                {"range": [0, 25],  "color": "#0d2d0d"},
                {"range": [25, 50], "color": "#2d2000"},
                {"range": [50, 75], "color": "#2d1200"},
                {"range": [75, 100],"color": "#2d0000"},
            ],
            "threshold": {"line": {"color": "white", "width": 2}, "value": 75}
        },
        title={"text": f"{row['machine_id']}<br><span style='font-size:12px'>RUL: {row['rul_hours']}h</span>",
               "font": {"color": "#e6edf3", "size": 14}}
    ))
    fig_g.update_layout(
        paper_bgcolor="#161b22", height=200,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    cols[i].plotly_chart(fig_g, use_container_width=True)

# ── Sensor Trends ─────────────────────────────────────────────────────────────
st.markdown("### 📈 Sensor Trend Analysis")
if not sensor_df.empty:
    machine_list = [localize_machine_name(f"M-{i:03d}") for i in sorted(sensor_df["machine_id"].unique())]
    sel_machine = st.selectbox("Select Machine", machine_list, key="machine_sel")
    
    # Reverse mapping to get original integer ID
    rev_mapping = {localize_machine_name(f"M-{i:03d}"): i for i in range(1, 100)}
    mid = rev_mapping.get(sel_machine, 1)
    
    mdf = sensor_df[sensor_df["machine_id"] == mid].tail(500)

    sensor_opt = st.selectbox("Sensor Channel", ["temperature","vibration","pressure","rpm","current"])
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=mdf["timestamp"], y=mdf[sensor_opt],
        mode="lines", name=sensor_opt,
        line=dict(color="#58a6ff", width=1.5),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.08)"
    ))
    # Highlight failure zones
    if "failure_in_24h" in mdf.columns:
        fail_mask = mdf["failure_in_24h"] == 1
        fig_trend.add_trace(go.Scatter(
            x=mdf.loc[fail_mask, "timestamp"], y=mdf.loc[fail_mask, sensor_opt],
            mode="markers", name="Failure Zone",
            marker=dict(color="#ff4d4f", size=4, symbol="x")
        ))
    fig_trend.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3"), height=300,
        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
        margin=dict(l=40, r=20, t=20, b=40), legend=dict(bgcolor="#161b22")
    )
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("Run `python models/lstm_maintenance.py` to generate sensor data and train the LSTM model.")

# ── Detailed Table ────────────────────────────────────────────────────────────
st.markdown("### 📋 Full Machine Status Table")
def sev_badge(s):
    colors = {"CRITICAL":"#ff4d4f","HIGH":"#ff7a00","MEDIUM":"#fadb14","LOW":"#52c41a"}
    return f"<span style='color:{colors.get(s,\"#aaa\")};font-weight:700'>{s}</span>"

display_df = df.copy()
st.dataframe(
    display_df.style.background_gradient(subset=["health_score"], cmap="RdYlGn")
               .background_gradient(subset=["failure_probability"], cmap="RdYlGn_r")
               .format({"failure_probability": "{:.1%}", "health_score": "{:.1f}"}),
    use_container_width=True, height=350
)

st.caption("🤖 Predictions generated by LSTM model trained on multi-variate sensor telemetry")
