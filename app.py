import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Industrial Predictive Maintenance & Downtime Cost Estimator",
    page_icon="⚙️",
    layout="wide"
)

# 2. Custom Styling
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.status-normal {
    background-color: #d4edda;
    color: #155724;
    padding: 12px;
    border-radius: 6px;
    font-weight: bold;
}
.status-warning {
    background-color: #f8d7da;
    color: #721c24;
    padding: 12px;
    border-radius: 6px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("⚙️ Industrial Predictive Maintenance Engine")
st.markdown("**Real-time Mechanical Telemetry Analysis & Financial Risk Optimization**")
st.markdown("---")

# 3. Artifact Loading with Case-Insensitive Fallbacks
@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        (os.path.join(BASE_DIR, "Artifacts", "model.pkl"), os.path.join(BASE_DIR, "Artifacts", "threshold.pkl")),
        (os.path.join(BASE_DIR, "artifacts", "model.pkl"), os.path.join(BASE_DIR, "artifacts", "threshold.pkl")),
        ("Artifacts/model.pkl", "Artifacts/threshold.pkl"),
        ("artifacts/model.pkl", "artifacts/threshold.pkl")
    ]
    
    for m_path, t_path in possible_paths:
        if os.path.exists(m_path) and os.path.exists(t_path):
            m = joblib.load(m_path)
            t = joblib.load(t_path)
            return m, t
            
    raise FileNotFoundError("Could not locate model.pkl and threshold.pkl in 'Artifacts/' or 'artifacts/'")

# Initialize variables to prevent NameError
model = None
optimal_threshold = 0.25
artifacts_loaded = False

try:
    model, optimal_threshold = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.error(f"⚠️ Could not load model artifacts. Details: {e}")

# 4. Sidebar Controls
st.sidebar.header("🎛️ Telemetry Sensor Inputs")
air_temp = st.sidebar.slider("Air Temperature (K)", min_value=290.0, max_value=310.0, value=300.0, step=0.1)
process_temp = st.sidebar.slider("Process Temperature (K)", min_value=300.0, max_value=320.0, value=310.0, step=0.1)
rotational_speed = st.sidebar.slider("Rotational Speed (RPM)", min_value=1000, max_value=2800, value=1500, step=10)
torque = st.sidebar.slider("Torque (Nm)", min_value=3.0, max_value=80.0, value=40.0, step=0.5)
tool_wear = st.sidebar.slider("Tool Wear (min)", min_value=0, max_value=250, value=100, step=1)

# 5. Physics Transformations
temp_diff = process_temp - air_temp
power_watts = torque * (rotational_speed * (2 * np.pi / 60))
overstrain_factor = torque * tool_wear

# 6. Main Layout
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("🔧 Derived Physics & Telemetry Metrics")
    
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Temp Difference (ΔT)", f"{temp_diff:.1f} K", delta="Normal" if temp_diff >= 8.6 else "Low Heat Dissipation", delta_color="normal" if temp_diff >= 8.6 else "inverse")
        st.metric("Mechanical Power", f"{power_watts / 1000.0:.2f} kW", delta="Normal Range" if 3500 <= power_watts <= 9000 else "Out of Bounds", delta_color="normal" if 3500 <= power_watts <= 9000 else "inverse")
        
    with m2:
        st.metric("Overstrain Load", f"{overstrain_factor:,.0f} Nm·min", delta="High Load" if overstrain_factor > 11000 else "Safe Load", delta_color="inverse" if overstrain_factor > 11000 else "normal")
        st.metric("Current Tool Wear", f"{tool_wear} min", delta="High Wear" if tool_wear > 200 else "Normal Wear", delta_color="inverse" if tool_wear > 200 else "normal")

    st.markdown("---")
    st.markdown("### 💰 Financial Risk Model Parameters")
    st.markdown("- **Preventive Maintenance Cost:** `$1,200` (Scheduled Inspection & Replacement)")
    st.markdown("- **Unscheduled Failure Cost:** `$10,000` (Production Downtime + Major Damage)")
    st.markdown(f"- **Cost-Optimized Failure Decision Threshold:** `{optimal_threshold:.2f}`")

with col_right:
    st.subheader("🚨 Risk Assessment & Prediction")
    
    if artifacts_loaded:
        input_data = pd.DataFrame([{
            'air_temp': air_temp,
            'process_temp': process_temp,
            'temp_diff': temp_diff,
            'rotational_speed': rotational_speed,
            'torque': torque,
            'power_watts': power_watts,
            'tool_wear': tool_wear,
            'overstrain_factor': overstrain_factor
        }])
        
        failure_prob = model.predict_proba(input_data)[0][1]
        is_failure_predicted = failure_prob >= optimal_threshold
        
        st.markdown(f"#### Calculated Failure Probability: **{failure_prob * 100:.2f}%**")
        st.progress(float(failure_prob))
        
        if is_failure_predicted:
            st.markdown(f"""
            <div class="status-warning">
                ⚠️ WARNING: HIGH FAILURE RISK DETECTED!<br>
                Probability ({failure_prob * 100:.1f}%) exceeds cost-optimized threshold ({optimal_threshold:.2f}).
            </div>
            """, unsafe_allow_html=True)
            
            st.warning("⚠️ **Recommended Action:** Schedule Immediate Preventive Maintenance ($1,200).")
            
            st.markdown("#### 💸 Financial Loss Matrix Analysis")
            st.error(f"""
            - **If Action Taken (Preventive Maintenance):** Costs **$1,200**
            - **If Ignored (Unscheduled Breakdown Risk):** Potential Loss **$10,000**
            - **Net Risk Mitigation Value:** **${10000 - 1200:,.0f} Saved**
            """)
        else:
            st.markdown(f"""
            <div class="status-normal">
                ✅ NORMAL OPERATIONAL STATUS<br>
                Probability ({failure_prob * 100:.1f}%) is within safe limits (below {optimal_threshold:.2f} threshold).
            </div>
            """, unsafe_allow_html=True)
            
            st.success("🟢 **Recommended Action:** Continue standard operations. No immediate intervention required.")
            st.info("Expected Operating Cost: **$0** (No maintenance overhead required)")
    else:
        st.warning("Model artifacts are not loaded. Please ensure `Artifacts/model.pkl` and `Artifacts/threshold.pkl` are present in your repo.")

    st.markdown("---")
    st.markdown("### 🔍 Root Cause Risk Factors")
    risk_triggers = []
    if temp_diff < 8.6:
        risk_triggers.append("• **Heat Dissipation Risk:** Temperature difference ΔT < 8.6 K.")
    if overstrain_factor > 11000:
        risk_triggers.append("• **Overstrain Stress:** Combined torque and tool wear exceed shear thresholds.")
    if power_watts < 3500 or power_watts > 9000:
        risk_triggers.append("• **Power Out-of-Bounds:** Mechanical power output outside safe 3.5 kW - 9.0 kW window.")
    if tool_wear > 200:
        risk_triggers.append("• **Critical Tool Degradation:** Tool wear exceeds 200 minutes.")
        
    if risk_triggers:
        for trigger in risk_triggers:
            st.write(trigger)
    else:
        st.write("• All primary sensor telemetry indicators are operating within normal engineering limits.")
        