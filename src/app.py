import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

from src.database.lims_db import init_lims_db, get_lims_dataframe
from src.models.simulator import FormulationSimulator
from src.agent.lab_copilot import LabCopilotAgent

# Page configuration
st.set_page_config(
    page_title="FormuAI — FMCG R&D Lab Intelligence Platform",
    page_icon="🔬",
    layout="wide"
)

# Custom Premium Styling (Vanilla CSS injected)
st.markdown("""
<style>
    /* Import modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Space+Grotesk:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: #0F172A;
    }
    
    /* Top Banner Gradient */
    .top-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #0369A1 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .top-banner h1 {
        color: #F8FAFC !important;
        margin: 0;
        font-weight: 700;
        font-size: 2.5rem;
    }
    
    .top-banner p {
        color: #BAE6FD;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0284C7;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database and load Simulator / Agent singletons
@st.cache_resource
def load_application_resources():
    db_path = init_lims_db()
    simulator = FormulationSimulator()
    simulator.load_model()
    agent = LabCopilotAgent()
    return db_path, simulator, agent

db_path, simulator, agent = load_application_resources()

# Top Banner UI
st.markdown("""
<div class="top-banner">
    <h1>🔬 FormuAI</h1>
    <p>AI-Powered Formulation Optimizer & Digital Lab Integration System for FMCG R&D</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/chemistry.png", width=70)
st.sidebar.markdown("### **R&D Lab Navigation**")
tab_selection = st.sidebar.radio(
    "Go To Module:",
    ["🔬 Formulation Simulator", "💬 AI Lab Copilot (Copilot Studio)", "📊 LIMS Analytics Dashboard"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Domain context:**
- *Business Group:* Personal Care R&D
- *Core formulation:* Conditioning shampoo
- *Mock Server:* Connected to LIMS SQL db
""")

# ----------------- MODULE 1: FORMULATION SIMULATION -----------------
if tab_selection == "🔬 Formulation Simulator":
    st.subheader("🔬 AI-Driven Formulation Simulator & Optimizer")
    st.write("Adjust raw ingredient ratios and processing settings to run virtual lab tests instantaneously. Minimizes physical trial-and-error.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### **Input Parameters**")
        st.markdown("#### *Ingredient Ratios (% w/w)*")
        
        surfactant = st.slider("Primary Surfactant (SLES)", 8.0, 16.0, 12.0, 0.1)
        co_surfactant = st.slider("Co-Surfactant (CAPB)", 1.5, 5.5, 3.5, 0.1)
        thickener = st.slider("Thickener (NaCl)", 0.1, 3.0, 1.2, 0.1)
        active = st.slider("Active Ingredient (Silicones)", 0.1, 2.5, 1.0, 0.1)
        
        # Calculate water automatically to make it sum to 100%
        water = round(100.0 - (surfactant + co_surfactant + thickener + active), 2)
        st.info(f"💧 **Water (Balance):** {water}% w/w")
        
        st.markdown("#### *Process parameters*")
        temp = st.slider("Mixing Temperature (°C)", 40.0, 85.0, 60.0, 0.5)
        shear = st.slider("Shear Rate (RPM)", 500, 2500, 1200, 50)
        
        inputs = {
            'surfactant_pct': surfactant,
            'co_surfactant_pct': co_surfactant,
            'thickener_pct': thickener,
            'active_ingredient_pct': active,
            'water_pct': water,
            'mixing_temp_c': temp,
            'shear_rate_rpm': float(shear)
        }
        
    with col2:
        st.markdown("### **Simulation Outputs (Physical Properties)**")
        
        # Get Predictions
        preds = simulator.predict(inputs)
        
        pcol1, pcol2 = st.columns(2)
        
        with pcol1:
            # Viscosity Gauge
            fig_visc = go.Figure(go.Indicator(
                mode="gauge+number",
                value=preds['viscosity_cp'],
                title={'text': "Viscosity (cP)"},
                gauge={'axis': {'range': [0, 8000]},
                       'bar': {'color': "#0284C7"},
                       'steps': [
                           {'range': [0, 2500], 'color': "#fee2e2"},
                           {'range': [2500, 5500], 'color': "#dcfce7"},
                           {'range': [5500, 8000], 'color': "#fef9c3"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 7000}}
            ))
            fig_visc.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_visc, use_container_width=True)
            
            # pH Gauge
            fig_ph = go.Figure(go.Indicator(
                mode="gauge+number",
                value=preds['ph'],
                title={'text': "pH Level"},
                gauge={'axis': {'range': [3, 9]},
                       'bar': {'color': "#059669"},
                       'steps': [
                           {'range': [3, 5.5], 'color': "#fef9c3"},
                           {'range': [5.5, 7.5], 'color': "#dcfce7"},
                           {'range': [7.5, 9], 'color': "#fee2e2"}]}
            ))
            fig_ph.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_ph, use_container_width=True)
            
        with pcol2:
            # Stability Gauge
            fig_stab = go.Figure(go.Indicator(
                mode="gauge+number",
                value=preds['stability_days'],
                title={'text': "Emulsion Stability (Days)"},
                gauge={'axis': {'range': [0, 365]},
                       'bar': {'color': "#8B5CF6"},
                       'steps': [
                           {'range': [0, 90], 'color': "#fee2e2"},
                           {'range': [90, 180], 'color': "#fef9c3"},
                           {'range': [180, 365], 'color': "#dcfce7"}]}
            ))
            fig_stab.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_stab, use_container_width=True)
            
            # Foam Volume Gauge
            fig_foam = go.Figure(go.Indicator(
                mode="gauge+number",
                value=preds['foam_volume_ml'],
                title={'text': "Foam Volume (mL)"},
                gauge={'axis': {'range': [0, 800]},
                       'bar': {'color': "#F59E0B"},
                       'steps': [
                           {'range': [0, 250], 'color': "#fee2e2"},
                           {'range': [250, 500], 'color': "#fef9c3"},
                           {'range': [500, 800], 'color': "#dcfce7"}]}
            ))
            fig_foam.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_foam, use_container_width=True)

    # Formulation Design Optimizer Section
    st.markdown("---")
    st.markdown("### 🧬 AI-Powered Formulation Optimizer (Target Search)")
    st.write("Input your desired physical specifications, and the optimizer will run thousands of virtual iterations to propose the optimal ingredient formula.")
    
    ocol1, ocol2 = st.columns(2)
    with ocol1:
        target_visc = st.number_input("Target Viscosity (cP)", min_value=500.0, max_value=7000.0, value=3500.0, step=100.0)
    with ocol2:
        target_ph = st.number_input("Target pH", min_value=4.5, max_value=8.0, value=5.5, step=0.1)
        
    if st.button("🧬 Optimize Formulation"):
        opt_inputs, opt_outputs = simulator.optimize_formulation(target_visc, target_ph)
        
        ocol_a, ocol_b = st.columns(2)
        with ocol_a:
            st.success("Proposed Chemical Formulation:")
            opt_df = pd.DataFrame([{
                "Primary Surfactant": f"{opt_inputs['surfactant_pct']}%",
                "Co-Surfactant": f"{opt_inputs['co_surfactant_pct']}%",
                "Thickener (Salt)": f"{opt_inputs['thickener_pct']}%",
                "Active Ingredient": f"{opt_inputs['active_ingredient_pct']}%",
                "Water (Balance)": f"{opt_inputs['water_pct']}%",
                "Mixing Temp": f"{opt_inputs['mixing_temp_c']}°C",
                "Shear RPM": f"{opt_inputs['shear_rate_rpm']} RPM"
            }]).T
            opt_df.columns = ["Value"]
            st.table(opt_df)
            
        with ocol_b:
            st.info("Predicted Properties for Proposed Formula:")
            pred_df = pd.DataFrame([{
                "Predicted Viscosity": f"{opt_outputs['viscosity_cp']:.1f} cP (Target: {target_visc})",
                "Predicted pH": f"{opt_outputs['ph']:.2f} (Target: {target_ph})",
                "Predicted Stability": f"{opt_outputs['stability_days']:.0f} days",
                "Predicted Foam Volume": f"{opt_outputs['foam_volume_ml']:.1f} mL"
            }]).T
            pred_df.columns = ["Simulation Prediction"]
            st.table(pred_df)

# ----------------- MODULE 2: AI LAB COPILOT -----------------
elif tab_selection == "💬 AI Lab Copilot (Copilot Studio)":
    st.subheader("💬 AI Lab Copilot Agent Simulator")
    st.write("A mock version of a Microsoft Copilot Studio agent. Ask questions to query LIMS databases, predict parameters, or design formulations.")
    
    # Simple chat thread storage
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your FormuAI Lab Assistant. I can search past LIMS trials, run virtual tests, or optimize a recipe for you. Try typing: *'Show experiments by Madhura'* or *'Design a formulation with viscosity 4000 and pH 5.5'*" }
        ]
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if user_query := st.chat_input("Enter your R&D instruction..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Process through agent router
        agent_res = agent.process_prompt(user_query)
        
        # Format response based on routing output
        with st.chat_message("assistant"):
            st.write(agent_res["message"])
            
            if agent_res["intent"] == "lims_query":
                st.code(f"SQL QUERY EXECUTED:\n{agent_res['sql_executed']}", language="sql")
                if agent_res["records"]:
                    st.dataframe(pd.DataFrame(agent_res["records"]))
                else:
                    st.warning("No records found in database matching those search criteria.")
                    
            elif agent_res["intent"] in ["predict", "design"]:
                col_i, col_o = st.columns(2)
                with col_i:
                    st.markdown("**Formulation parameters:**")
                    st.json(agent_res["composition"])
                with col_o:
                    st.markdown("**Predicted Quality outputs:**")
                    st.json(agent_res["predicted_properties"])
                    
            # Save assistant response
            st.session_state.messages.append({"role": "assistant", "content": agent_res["message"]})

# ----------------- MODULE 3: LIMS ANALYTICS -----------------
else:
    st.subheader("📊 LIMS Analytics Dashboard")
    st.write("Historical experimental statistics compiled from LIMS database trials (simulating a Power BI layout).")
    
    # Load dataset
    df = get_lims_dataframe(db_path)
    
    # 1. Executive Summary Indicators
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Lab Trials</div>
            <div class="metric-value">{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Viscosity</div>
            <div class="metric-value">{df['viscosity_cp'].mean():.0f} cP</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Emulsion Stability</div>
            <div class="metric-value">{df['stability_days'].mean():.0f} days</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Primary Researchers</div>
            <div class="metric-value">{df['researcher_name'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # 2. Charts and Data Visualizations
    ccol1, ccol2 = st.columns(2)
    
    with ccol1:
        st.markdown("#### **Viscosity vs. Thickener Concentration (with Temperature shading)**")
        fig_scatter = px.scatter(
            df,
            x="thickener_pct",
            y="viscosity_cp",
            color="mixing_temp_c",
            size="surfactant_pct",
            labels={
                "thickener_pct": "Thickener Salt (% w/w)",
                "viscosity_cp": "Measured Viscosity (cP)",
                "mixing_temp_c": "Mixing Temp (°C)"
            },
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with ccol2:
        st.markdown("#### **Emulsion Stability Distribution**")
        fig_hist = px.histogram(
            df,
            x="stability_days",
            nbins=20,
            color_discrete_sequence=["#8B5CF6"],
            labels={"stability_days": "Stability Duration (Days)"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    # Correlation analysis
    st.markdown("#### **Ingredient Correlations (Targeting physical properties)**")
    numeric_cols = ['surfactant_pct', 'co_surfactant_pct', 'thickener_pct', 'active_ingredient_pct', 'mixing_temp_c', 'viscosity_cp', 'ph', 'stability_days', 'foam_volume_ml']
    corr = df[numeric_cols].corr()
    
    fig_heatmap = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
