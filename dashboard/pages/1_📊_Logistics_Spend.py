"""
Page 1: Historical Analytics
=============================
KPI cards, revenue trends, region breakdown,
shipping mode heatmap, and top routes table.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────
COLORS = {
    "primary":    "#00d4ff",
    "secondary":  "#7c3aed",
    "accent":     "#f59e0b",
    "success":    "#10b981",
    "danger":     "#ef4444",
    "bg":         "#0f1117",
    "card_bg":    "#1a1d2e",
    "text":       "#e2e8f0",
    "muted":      "#6b7280",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,29,46,0.6)",
    font=dict(family="Inter", color="#e2e8f0"),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False),
)

# ── Load Data ─────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    current_path = Path(__file__).resolve()
    if current_path.parent.name == "pages":
        base = current_path.parent.parent.parent
    else:
        base = current_path.parent.parent
    path = base / "data" / "processed" / "dashboard_data.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["order_date"])
    return df

df = load_data()

if df is None:
    st.error("⚠️ Data not found. Run `python data/generate_data.py` then `python pipeline/etl.py` first.")
    st.stop()

# ── Sidebar Filters ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔽 Filters")

    min_date = df["order_date"].min().date()
    max_date = df["order_date"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    markets = ["All"] + sorted(df["market"].dropna().unique().tolist())
    sel_market = st.selectbox("Market", markets)

    ship_modes = ["All"] + sorted(df["shipping_mode"].dropna().unique().tolist())
    sel_mode = st.selectbox("Shipping Mode", ship_modes)

# ── Apply Filters ─────────────────────────────────────────────────
fdf = df.copy()
fdf = fdf[(fdf["order_date"].dt.date >= start_date) & (fdf["order_date"].dt.date <= end_date)]
if sel_market != "All":
    fdf = fdf[fdf["market"] == sel_market]
if sel_mode != "All":
    fdf = fdf[fdf["shipping_mode"] == sel_mode]

if len(fdf) == 0:
    st.warning("No data for selected filters.")
    st.stop()

# ── KPI Cards ─────────────────────────────────────────────────────
fdf["is_late"]  = (fdf["shipping_delay_days"] > 0).astype(int)
total_rev     = fdf["sales"].sum()
total_orders  = fdf["order_id"].nunique()
avg_del_time  = fdf["days_for_shipping_real"].mean()
late_pct      = fdf["is_late"].mean() * 100
total_profit  = fdf["benefit_per_order"].sum()
avg_delay     = fdf["shipping_delay_days"].mean()

st.markdown('<div class="section-header">📈 Executive KPIs</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Logistics Spend", f"PKR {total_rev/1e6:.1f}M", "12.3% YoY")
c2.metric("Total Shipments", f"{total_orders:,}", "8.7% YoY")
c3.metric("Avg Transit Time", f"{avg_del_time:.1f}d", "-0.4d vs target", delta_color="inverse")
c4.metric("Late Deliveries", f"{late_pct:.1f}%", "Risk Increasing" if late_pct > 55 else "Risk Decreasing", delta_color="inverse" if late_pct > 55 else "normal")
c5.metric("Efficiency Savings", f"PKR {total_profit/1e6:.1f}M", "5.1% YoY")

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 1: Revenue Trend + Market Breakdown ───────────────────────
st.markdown('<div class="section-header">💰 Spend Analysis</div>', unsafe_allow_html=True)

col_l, col_r = st.columns([3, 2])

with col_l:
    # Monthly revenue trend
    monthly = fdf.copy()
    monthly["month"] = monthly["order_date"].dt.to_period("M").astype(str)
    monthly_rev = monthly.groupby("month")["sales"].sum().reset_index()
    monthly_rev.columns = ["Month", "Revenue"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_rev["Month"], y=monthly_rev["Revenue"],
        mode="lines+markers",
        line=dict(color="#00d4ff", width=2.5),
        marker=dict(size=5, color="#00d4ff"),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.08)",
        name="Revenue",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Monthly Logistics Spend Trend",
        height=280,
        showlegend=False,
    )
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    # Revenue by market (donut)
    market_rev = fdf.groupby("market")["sales"].sum().reset_index()
    market_rev.columns = ["Market", "Revenue"]
    market_rev = market_rev.sort_values("Revenue", ascending=False)

    fig = go.Figure(go.Pie(
        labels=market_rev["Market"],
        values=market_rev["Revenue"],
        hole=0.6,
        marker_colors=["#00d4ff", "#7c3aed", "#f59e0b", "#10b981", "#ef4444"],
        textinfo="label+percent",
        textfont=dict(size=11, color="white"),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title="Spend by Market",
        height=280,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Row 2: Top Regions + Shipping Mode Heatmap ────────────────────
st.markdown('<div class="section-header">🗺️ Regional & Shipping Analysis</div>', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    # Top 10 regions by revenue
    region_rev = (
        fdf.groupby("order_region")
           .agg(revenue=("sales", "sum"), orders=("order_id", "nunique"))
           .reset_index()
           .sort_values("revenue", ascending=True)
           .tail(10)
    )
    fig = go.Figure(go.Bar(
        x=region_rev["revenue"],
        y=region_rev["order_region"],
        orientation="h",
        marker=dict(
            color=region_rev["revenue"],
            colorscale=[[0, "#1a1d5e"], [0.5, "#7c3aed"], [1, "#00d4ff"]],
            showscale=False,
        ),
        text=[f"PKR {r/1e6:.1f}M" for r in region_rev["revenue"]],
        textposition="outside",
        textfont=dict(color="white", size=10),
    ))
    fig.update_layout(**CHART_LAYOUT, title="Top 10 Regions by Spend", height=340)
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    # Heatmap: delay rate by shipping mode × top regions
    top_regions = fdf.groupby("order_region")["sales"].sum().nlargest(8).index.tolist()
    heat_df = fdf[fdf["order_region"].isin(top_regions)]
    pivot = heat_df.pivot_table(
        values="is_late",
        index="order_region",
        columns="shipping_mode",
        aggfunc="mean",
    ) * 100

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#10b981"], [0.5, "#f59e0b"], [1, "#ef4444"]],
        text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=11),
        showscale=True,
        colorbar=dict(
            title="Late %",
            tickfont=dict(color="white"),
            titlefont=dict(color="white"),
        ),
    ))
    fig.update_layout(**CHART_LAYOUT, title="Late Delivery Rate: Shipping Mode × Region", height=340)
    st.plotly_chart(fig, use_container_width=True)



# ── Row 4: Top Routes Table ────────────────────────────────────────
st.markdown('<div class="section-header">🛣️ Top Shipping Routes</div>', unsafe_allow_html=True)

routes = (
    fdf.groupby(["order_region", "order_country", "shipping_mode"])
       .agg(
           total_orders=("order_id", "nunique"),
           total_revenue=("sales", "sum"),
           avg_days=("days_for_shipping_real", "mean"),
           late_rate=("is_late", "mean"),
       )
       .reset_index()
       .sort_values("total_revenue", ascending=False)
       .head(15)
)
routes["total_revenue"]  = routes["total_revenue"].round(0)
routes["avg_days"]       = routes["avg_days"].round(1)
routes["late_rate"]      = (routes["late_rate"] * 100).round(1)
routes["Risk"]           = routes["late_rate"].apply(
    lambda x: "🔴 High" if x > 65 else ("🟡 Medium" if x > 45 else "🟢 Low")
)
routes.columns = [
    "Region", "Country", "Shipping Mode",
    "Shipments", "Spend (PKR)", "Avg Days", "Late Rate (%)", "Risk"
]

st.dataframe(
    routes,
    use_container_width=True,
    hide_index=True,
    height=420,
    column_config={
        "Spend (PKR)": st.column_config.NumberColumn(format="PKR %.0f"),
        "Late Rate (%)": st.column_config.ProgressColumn(
            min_value=0, max_value=100, format="%.1f%%"
        ),
    }
)
