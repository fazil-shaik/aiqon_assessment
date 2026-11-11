"""
Aiqon AI Logistics Platform - Shipment ETA Prediction & Delay Propagation Engine
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.delay_propagation import DelayPropagationEngine, train_and_cache_model
from models.explainability import DelayExplainer
from models.evaluation import evaluate_models_and_baselines
from data.generator import generate_shipment_dataset, ROUTES, CARRIERS, CUSTOMS_LANES

# Streamlit Page Config
st.set_page_config(
    page_title="Aiqon | AI Shipment ETA & Delay Propagation Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0284C7;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .diagnostic-box {
        background: #F0FDF4;
        border-left: 5px solid #10B981;
        padding: 14px 18px;
        border-radius: 4px;
        font-size: 0.95rem;
        margin-bottom: 12px;
    }
    .diagnostic-box-alert {
        background: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 14px 18px;
        border-radius: 4px;
        font-size: 0.95rem;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
        padding: 10px 18px;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading and model training
@st.cache_data
def load_data():
    csv_path = os.path.join("data", "shipments_dataset.csv")
    if not os.path.exists(csv_path):
        df = generate_shipment_dataset(n_samples=500, random_seed=42)
        os.makedirs("data", exist_ok=True)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
    return df

@st.cache_resource
def load_engine(df):
    engine = DelayPropagationEngine()
    engine.fit(df)
    return engine

@st.cache_data
def get_benchmarks(df):
    return evaluate_models_and_baselines(df)

# Initialize Session Data
df_shipments = load_data()
engine = load_engine(df_shipments)
explainer = DelayExplainer(engine)

# Header Section
col_brand1, col_brand2 = st.columns([0.75, 0.25])
with col_brand1:
    st.markdown('<div class="main-header">Aiqon | AI Shipment ETA & Delay Propagation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Stage Maritime Logistics Delay Cascading, Dynamic In-Flight Inference & Root-Cause Explainability</div>', unsafe_allow_html=True)
with col_brand2:
    st.markdown("""
        <div style="text-align: right; padding-top: 10px;">
            <span style="background-color: #E0F2FE; color: #0369A1; padding: 6px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem;">
                Production Model Online
            </span>
        </div>
    """, unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.markdown("### **Fleet Control Tower**")
    st.markdown("Operational filters and simulation parameters.")
    
    selected_carrier_filter = st.selectbox(
        "Filter by Ocean Carrier",
        ["All Carriers"] + sorted(list(df_shipments["carrier"].unique()))
    )
    
    selected_route_filter = st.selectbox(
        "Filter by Maritime Trade Lane",
        ["All Routes"] + list(ROUTES.keys()),
        format_func=lambda x: "All Routes" if x == "All Routes" else f"{x}: {ROUTES[x]['origin'].split()[0]} to {ROUTES[x]['destination'].split()[0]}"
    )
    
    sla_threshold_hours = st.slider("Target SLA Delay Threshold (Hours)", min_value=1.0, max_value=24.0, value=6.0, step=1.0)
    
    st.markdown("---")
    st.markdown("#### **System Architecture**")
    st.markdown("""
    - **Backbone:** Gradient Boosted Stage Cascades
    - **Propagation:** Berth Penalty & Buffer Multipliers
    - **Inference:** In-Flight Dynamic Milestones (0-3)
    - **Explainability:** Local Factor Attribution & Diagnostic Playbooks
    """)

# Apply Sidebar Filters to active view dataset
filtered_df = df_shipments.copy()
if selected_carrier_filter != "All Carriers":
    filtered_df = filtered_df[filtered_df["carrier"] == selected_carrier_filter]
if selected_route_filter != "All Routes":
    filtered_df = filtered_df[filtered_df["route_id"] == selected_route_filter]

# Main Tab Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Fleet Operations Dashboard",
    "Live Journey & In-Flight ETA Predictor",
    "Delay Propagation & Root-Cause AI",
    "Model Benchmark & Evaluation",
    "Dataset Explorer & Synthetic Simulator"
])

# -------------------------------------------------------------
# TAB 1: FLEET OPERATIONS DASHBOARD
# -------------------------------------------------------------
with tab1:
    st.markdown("### Global Fleet Visibility & Delay Propagation Health")
    st.markdown("Real-time monitoring across global shipping corridors, terminal congestion bottlenecks, and delivery variance.")
    
    # KPI Summary Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    total_active = len(filtered_df)
    delayed_count = int((filtered_df["total_delay_hours"] > sla_threshold_hours).sum())
    on_time_pct = ((total_active - delayed_count) / max(1, total_active)) * 100
    avg_delay_hrs = filtered_df["total_delay_hours"].mean()
    berth_miss_pct = (filtered_df["berth_window_missed"].mean()) * 100
    severe_delays = (filtered_df["total_delay_hours"] > 24.0).sum()
    
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tracked Journeys</div>
            <div class="metric-value">{total_active}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">On-Time SLA Rate</div>
            <div class="metric-value" style="color: {'#10B981' if on_time_pct >= 60 else '#EF4444'};">{on_time_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Arrival Variance</div>
            <div class="metric-value" style="color: {'#F59E0B' if avg_delay_hrs > 0 else '#10B981'};">{avg_delay_hrs:+.1f}h</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Berth Window Misses</div>
            <div class="metric-value" style="color: #EF4444;">{berth_miss_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Severe Delays (>24h)</div>
            <div class="metric-value" style="color: #6366F1;">{severe_delays}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualizations Row 1
    row1_col1, row1_col2 = st.columns([0.6, 0.4])
    
    with row1_col1:
        st.markdown("#### **Shipment Delay Distribution Across Fleet**")
        fig_dist = px.histogram(
            filtered_df,
            x="total_delay_hours",
            color="is_delayed",
            nbins=40,
            labels={"total_delay_hours": "Arrival Delay (Hours)", "is_delayed": f"Delayed (> {sla_threshold_hours}h)"},
            color_discrete_map={0: "#10B981", 1: "#EF4444"},
            marginal="box",
            opacity=0.85
        )
        fig_dist.add_vline(x=0, line_dash="dash", line_color="#475569", annotation_text="Scheduled ETA Baseline")
        fig_dist.add_vline(x=sla_threshold_hours, line_dash="dot", line_color="#EF4444", annotation_text=f"SLA Breach (+{sla_threshold_hours}h)")
        fig_dist.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with row1_col2:
        st.markdown("#### **Primary Root Causes of Delay**")
        cause_counts = filtered_df["primary_root_cause"].value_counts().reset_index()
        cause_counts.columns = ["Root Cause", "Shipments"]
        fig_pie = px.pie(
            cause_counts,
            names="Root Cause",
            values="Shipments",
            color="Root Cause",
            color_discrete_map={
                "Port Congestion/Berth": "#EF4444",
                "Customs Clearance": "#F59E0B",
                "Origin Delay": "#3B82F6",
                "Inland Delivery": "#8B5CF6",
                "Transit/Weather": "#06B6D4"
            },
            hole=0.45
        )
        fig_pie.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Visualizations Row 2
    row2_col1, row2_col2 = st.columns([0.5, 0.5])
    
    with row2_col1:
        st.markdown("#### **Delay Variance by Maritime Trade Corridor**")
        route_perf = filtered_df.groupby("route_id").agg({
            "total_delay_hours": ["mean", "median", "std"],
            "berth_window_missed": "mean"
        }).reset_index()
        route_perf.columns = ["Route", "Mean_Delay", "Median_Delay", "Std_Delay", "Berth_Miss_Rate"]
        route_perf["Route_Name"] = route_perf["Route"].apply(lambda r: f"{ROUTES[r]['origin'].split()[0]} to {ROUTES[r]['destination'].split()[0]}")
        
        fig_route = px.bar(
            route_perf,
            x="Route_Name",
            y="Mean_Delay",
            error_y="Std_Delay",
            color="Berth_Miss_Rate",
            color_continuous_scale="Reds",
            labels={"Mean_Delay": "Mean Delay (Hours)", "Route_Name": "Trade Lane", "Berth_Miss_Rate": "Berth Miss Rate"},
        )
        fig_route.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_route, use_container_width=True)
        
    with row2_col2:
        st.markdown("#### **Non-Linear Dynamics: Berth Miss Delay Amplification**")
        fig_scatter = px.scatter(
            filtered_df,
            x="dest_port_congestion_idx",
            y="total_delay_hours",
            color="berth_window_missed",
            size="sched_transit_hours",
            color_discrete_map={0: "#3B82F6", 1: "#EF4444"},
            labels={
                "dest_port_congestion_idx": "Destination Port Congestion Index",
                "total_delay_hours": "Final Net Delay (Hours)",
                "berth_window_missed": "Missed Berthing Window"
            },
            hover_data=["shipment_id", "carrier", "route_id"]
        )
        fig_scatter.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: LIVE JOURNEY & IN-FLIGHT ETA PREDICTOR
# -------------------------------------------------------------
with tab2:
    st.markdown("### Live Shipment In-Flight ETA Predictor")
    st.markdown(
        "Simulate in-flight journeys across operational milestones. Observe how ETA and delay risks "
        "dynamically update as real-world stage milestones are achieved or external disruptions arise."
    )
    
    mode = st.radio("Choose Mode:", ["Select Existing Shipment from Fleet", "Custom In-Flight Scenario Builder"], horizontal=True)
    
    if mode == "Select Existing Shipment from Fleet":
        selected_shp_id = st.selectbox(
            "Select Shipment ID to inspect:",
            filtered_df["shipment_id"].tolist(),
            index=0
        )
        active_record = filtered_df[filtered_df["shipment_id"] == selected_shp_id].iloc[0].to_dict()
    else:
        st.markdown("#### Custom Scenario Parameters")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sel_route = st.selectbox("Trade Lane", list(ROUTES.keys()))
            sel_carrier = st.selectbox("Carrier", CARRIERS)
        with c2:
            sel_priority = st.selectbox("Cargo Priority", ["Standard", "Expedited", "Perishable/Cold Chain"])
            sel_customs = st.selectbox("Customs Risk Channel", CUSTOMS_LANES)
        with c3:
            orig_cong = st.slider("Origin Congestion Index", 0.0, 1.0, 0.45, 0.05)
            trans_weather = st.slider("Transit Weather / Sea State Severity", 0.0, 1.0, 0.55, 0.05)
        with c4:
            dest_cong = st.slider("Destination Port Congestion Index", 0.0, 1.0, 0.70, 0.05)
            inland_traffic = st.slider("Inland Traffic Index", 0.0, 1.0, 0.50, 0.05)
            
        r_info = ROUTES[sel_route]
        now_dt = datetime.now()
        sched_arr_dt = now_dt + timedelta(hours=r_info["total_sched_h"])
        
        active_record = {
            "shipment_id": "SIM-CUSTOM-001",
            "route_id": sel_route,
            "origin_port": r_info["origin"],
            "destination_port": r_info["destination"],
            "hub_port": r_info["hub_port"],
            "carrier": sel_carrier,
            "cargo_priority": sel_priority,
            "distance_nm": r_info["distance_nm"],
            "sched_departure": now_dt.strftime("%Y-%m-%d %H:%M"),
            "sched_arrival": sched_arr_dt.strftime("%Y-%m-%d %H:%M"),
            "sched_transit_hours": r_info["sched_transit_h"],
            "sched_port_dwell_hours": r_info["sched_port_dwell_h"],
            "sched_inland_hours": r_info["sched_inland_h"],
            "total_sched_hours": r_info["total_sched_h"],
            "origin_congestion_idx": orig_cong,
            "transit_weather_severity": trans_weather,
            "dest_port_congestion_idx": dest_cong,
            "customs_risk_tier": sel_customs,
            "inland_traffic_idx": inland_traffic,
            "berth_window_missed": 1 if dest_cong > 0.65 and trans_weather > 0.5 else 0,
            "buffer_absorbed_hours": 3.0,
            "stage_origin_delay_hours": round(orig_cong * 12.0, 1),
            "stage_transit_delay_hours": round((trans_weather - 0.3) * 20.0, 1) if trans_weather > 0.3 else 0.0,
            "stage_port_waiting_hours": round(dest_cong * 24.0, 1),
            "stage_customs_dwell_hours": 48.0 if "Red" in sel_customs else (12.0 if "Yellow" in sel_customs else 2.0),
            "stage_inland_delay_hours": round(inland_traffic * 8.0, 1),
            "total_delay_hours": 0.0,
            "is_delayed": 1,
        }

    st.markdown("---")
    
    # Milestone Control Bar
    st.markdown("#### **Active In-Flight Milestone Tracker (Incomplete Data Simulator)**")
    st.caption("Logistics ETA predictions must dynamically adapt as journeys progress and intermediate uncertainty dissolves.")
    
    milestone_names = {
        0: "Milestone 0: Booking / Pre-Departure (No realized stage data)",
        1: "Milestone 1: Origin Departed (Origin lag realized)",
        2: "Milestone 2: Ocean Arrived / Outer Anchorage (Transit lag realized, Berth window determined)",
        3: "Milestone 3: Customs Cleared (Port dwell & customs duration realized)"
    }
    
    selected_milestone = st.radio(
        "Select Active Journey Progress Milestone:",
        options=[0, 1, 2, 3],
        format_func=lambda m: milestone_names[m],
        horizontal=False,
        index=3
    )
    
    # Run Prediction at chosen milestone
    prediction_result = engine.predict_shipment(active_record, milestone=selected_milestone)
    explanation_result = explainer.explain_shipment(active_record, milestone=selected_milestone)
    
    # Prediction Header Cards
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Scheduled Arrival</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #1E293B;">{prediction_result['scheduled_arrival']}</div>
        </div>
        """, unsafe_allow_html=True)
    with pcol2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">AI Predicted ETA</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: {prediction_result['badge_color']};">{prediction_result['predicted_eta']}</div>
        </div>
        """, unsafe_allow_html=True)
    with pcol3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Delay Magnitude</div>
            <div class="metric-value" style="color: {prediction_result['badge_color']};">{prediction_result['predicted_delay_hours']:+.1f} hrs</div>
        </div>
        """, unsafe_allow_html=True)
    with pcol4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Delay Probability</div>
            <div class="metric-value" style="color: {prediction_result['badge_color']};">{prediction_result['delay_probability']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Journey Milestone Progress Cards
    st.markdown("#### **Stage-by-Stage Journey Progression**")
    m_cols = st.columns(4)
    cascade_stages = prediction_result["stage_cascade"]
    
    stage_labels = ["Stage 1: Origin", "Stage 2: Transit", "Stage 3: Port & Customs", "Stage 4: Inland"]
    for idx, (m_col, stage_info) in enumerate(zip(m_cols, cascade_stages)):
        with m_col:
            is_active = (selected_milestone >= idx)
            status_text = "Realized" if selected_milestone > idx else ("In Progress" if selected_milestone == idx else "Forecasted")
            border_col = "#10B981" if selected_milestone > idx else ("#3B82F6" if selected_milestone == idx else "#CBD5E1")
            st.markdown(f"""
            <div style="border: 2px solid {border_col}; border-radius: 8px; padding: 12px; background: {'#F0FDF4' if selected_milestone > idx else '#FFFFFF'};">
                <div style="font-size: 0.85rem; font-weight: 600; color: #64748B;">{stage_labels[idx]}</div>
                <div style="font-size: 1.05rem; font-weight: 700; margin-top: 2px;">{stage_info['stage']}</div>
                <div style="font-size: 0.85rem; color: #64748B; margin-top: 4px;">Status: <b>{status_text}</b></div>
                <div style="font-size: 1.15rem; font-weight: 700; color: {'#EF4444' if stage_info['delay_hours'] > 4 else '#1E293B'}; margin-top: 6px;">
                    {stage_info['delay_hours']:+.1f} hrs variance
                </div>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 3: DELAY PROPAGATION & ROOT-CAUSE EXPLAINABILITY
# -------------------------------------------------------------
with tab3:
    st.markdown("### Root-Cause Attribution & Delay Cascade Dynamics")
    st.markdown(
        "Aiqon's AI deconstructs the shipment lifecycle into additive and compounding delay drivers. "
        "Understand why delays emerge, whether buffer absorption occurred, and what operational action to take."
    )
    
    # Diagnostic narrative display
    if prediction_result["predicted_delay_hours"] > 6.0:
        st.markdown(f'<div class="diagnostic-box-alert">{explanation_result["narrative"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="diagnostic-box">{explanation_result["narrative"]}</div>', unsafe_allow_html=True)
        
    exp_col1, exp_col2 = st.columns([0.55, 0.45])
    
    with exp_col1:
        st.markdown("#### **Delay Propagation Waterfall (Net Hours from Baseline)**")
        # Waterfall Chart
        wf_factors = ["Scheduled ETA Baseline"]
        wf_impacts = [0.0]
        
        for c in explanation_result["contributions"]:
            wf_factors.append(c["factor"])
            wf_impacts.append(c["impact_hours"])
            
        wf_factors.append("Final Net ETA Variance")
        wf_impacts.append(prediction_result["predicted_delay_hours"])
        
        fig_wf = go.Figure(go.Waterfall(
            name="Delay Propagation",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(explanation_result["contributions"]) + ["total"],
            x=wf_factors,
            textposition="outside",
            text=[f"{val:+.1f}h" if val != 0 else "0.0h" for val in wf_impacts],
            y=wf_impacts,
            connector={"line": {"color": "#94A3B8"}},
            decreasing={"marker": {"color": "#10B981"}},
            increasing={"marker": {"color": "#EF4444"}},
            totals={"marker": {"color": "#0284C7"}}
        ))
        fig_wf.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            waterfallgap=0.3
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    with exp_col2:
        st.markdown("#### **Relative Factor Attribution Magnitude**")
        contrib_df = pd.DataFrame(explanation_result["contributions"])
        contrib_df["abs_impact"] = contrib_df["impact_hours"].abs()
        contrib_df = contrib_df.sort_values(by="abs_impact", ascending=True)
        
        fig_bar = px.bar(
            contrib_df,
            x="impact_hours",
            y="factor",
            orientation="h",
            color="impact_hours",
            color_continuous_scale="RdYlGn_r",
            labels={"impact_hours": "Attributed Variance (Hours)", "factor": "Factor"}
        )
        fig_bar.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Operational Playbook Recommendations
    st.markdown("#### **Operational Playbook & Actionable Mitigation Interventions**")
    for rec in explanation_result["recommendations"]:
        st.markdown(f"- **Recommendation:** {rec}")

# -------------------------------------------------------------
# TAB 4: MODEL EVALUATION & BENCHMARK COMPARISON
# -------------------------------------------------------------
with tab4:
    st.markdown("### Model Benchmark & Quantitative Evaluation")
    st.markdown(
        "Demonstrating model superiority against traditional industry baselines. "
        "Evaluated on regression metrics (MAE, RMSE, R²) and classification metrics (Accuracy, F1, ROC-AUC)."
    )
    
    with st.spinner("Evaluating multi-fold cross-validation across all models..."):
        benchmarks = get_benchmarks(df_shipments)
        
    b_df = benchmarks["benchmark_df"]
    
    st.markdown("#### **Benchmark Comparison Table**")
    st.dataframe(
        b_df.style.highlight_min(subset=["MAE (Hours)", "RMSE (Hours)"], color="#DCFCE7")
                  .highlight_max(subset=["R2 Score", "Accuracy", "F1 Score", "ROC-AUC"], color="#DCFCE7"),
        use_container_width=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    eval_c1, eval_c2 = st.columns([0.5, 0.5])
    
    with eval_c1:
        st.markdown("#### **In-Flight Convergence: Prediction Error Shrinkage by Milestone**")
        st.caption("As shipments reach successive stages, intermediate uncertainty collapses, driving continuous MAE reduction.")
        m_df = pd.DataFrame(list(benchmarks["milestone_maes"].items()), columns=["Milestone", "MAE (Hours)"])
        m_df["Description"] = [
            "Milestone 0: Pre-Booking",
            "Milestone 1: Origin Departed",
            "Milestone 2: Ocean Arrived",
            "Milestone 3: Customs Cleared"
        ]
        
        fig_mae = px.line(
            m_df,
            x="Description",
            y="MAE (Hours)",
            markers=True,
            text="MAE (Hours)",
            labels={"MAE (Hours)": "Mean Absolute Error (Hours)", "Description": "Journey Milestone Stage"}
        )
        fig_mae.update_traces(textposition="top center", line=dict(color="#0284C7", width=3), marker=dict(size=10, color="#0F172A"))
        fig_mae.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_mae, use_container_width=True)

    with eval_c2:
        st.markdown("#### **Global Model Feature Importance Ranking**")
        fi_df = engine.get_feature_importances().head(10)
        fig_fi = px.bar(
            fi_df,
            x="Importance_Pct",
            y="Feature",
            orientation="h",
            labels={"Importance_Pct": "Importance (%)", "Feature": "Engineered Feature"},
            color="Importance_Pct",
            color_continuous_scale="Blues"
        )
        fig_fi.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)
        
    st.markdown("#### **Residual Analysis (Predicted vs Actual Delays)**")
    res_df = pd.DataFrame({
        "Actual Delay (Hours)": benchmarks["actual_y_reg"],
        "Predicted Delay (Hours)": benchmarks["cv_reg_preds"],
        "Route": df_shipments["route_id"]
    })
    
    fig_resid = px.scatter(
        res_df,
        x="Actual Delay (Hours)",
        y="Predicted Delay (Hours)",
        color="Route",
        opacity=0.75,
        hover_data=["Route"]
    )
    # Ideal 45 degree line
    min_val = min(res_df["Actual Delay (Hours)"].min(), res_df["Predicted Delay (Hours)"].min())
    max_val = max(res_df["Actual Delay (Hours)"].max(), res_df["Predicted Delay (Hours)"].max())
    fig_resid.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect Calibration (y=x)",
        line=dict(color="#EF4444", dash="dash")
    ))
    fig_resid.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_resid, use_container_width=True)

# -------------------------------------------------------------
# TAB 5: DATASET EXPLORER & SYNTHETIC SIMULATOR
# -------------------------------------------------------------
with tab5:
    st.markdown("### Dataset Explorer & Synthetic Simulator")
    st.markdown(
        "Aiqon's synthetic journey generator creates mathematically grounded multi-stage shipment histories "
        "faithfully simulating marine transit physics, stochastic dwell times, port berth window expirations, "
        "and customs inspection tails."
    )
    
    st.markdown("#### **Dataset Overview & Filtering**")
    st.dataframe(filtered_df, use_container_width=True, height=350)
    
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Current Dataset (CSV)",
        data=csv_bytes,
        file_name="aiqon_shipments_dataset.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    st.markdown("#### **Generate New Synthetic Dataset**")
    gen_c1, gen_c2, gen_c3 = st.columns(3)
    with gen_c1:
        new_samples = st.number_input("Number of Shipment Records", min_value=100, max_value=2000, value=500, step=100)
    with gen_c2:
        new_seed = st.number_input("Random Generator Seed", min_value=1, max_value=9999, value=42, step=1)
    with gen_c3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Generate & Retrain Engine"):
            with st.spinner("Generating new synthetic logistics journeys..."):
                new_df = generate_shipment_dataset(n_samples=new_samples, random_seed=new_seed)
                new_df.to_csv("data/shipments_dataset.csv", index=False)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success(f"Successfully generated {new_samples} journeys and retrained predictive engine!")
                st.rerun()

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
    "Aiqon AI Logistics Platform • Built for Take-Home Assignment: Predicting Shipment ETA and Delay Propagation • 2026"
    "</div>",
    unsafe_allow_html=True
)
