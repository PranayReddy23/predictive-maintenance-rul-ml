import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Predictive Maintenance - RUL Predictor",
    page_icon="🛠️",
    layout="wide"
)

# =========================================================
# Custom Styling
# =========================================================
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
.metric-card {
    background-color: #111827;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #1f2937;
    box-shadow: 0 4px 14px rgba(0,0,0,0.18);
}
.metric-title {
    font-size: 0.95rem;
    color: #9ca3af;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f9fafb;
}
.small-note {
    color: #9ca3af;
    font-size: 0.92rem;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    margin-top: 0.4rem;
    margin-bottom: 0.6rem;
}
.status-good {
    color: #10b981;
    font-weight: 700;
}
.status-warn {
    color: #f59e0b;
    font-weight: 700;
}
.status-critical {
    color: #ef4444;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# File Paths
# =========================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_rf.pkl"
COLUMNS_PATH = BASE_DIR / "model_columns.pkl"
CAP_LIMITS_PATH = BASE_DIR / "cap_limits.pkl"
CATEGORY_VALUES_PATH = BASE_DIR / "category_values.pkl"
INPUT_RANGES_PATH = BASE_DIR / "input_ranges.pkl"

# =========================================================
# Helpers
# =========================================================
@st.cache_resource
def load_artifacts():
    required_files = [
        MODEL_PATH,
        COLUMNS_PATH,
        CAP_LIMITS_PATH,
        CATEGORY_VALUES_PATH,
        INPUT_RANGES_PATH
    ]

    missing = [str(f) for f in required_files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(COLUMNS_PATH, "rb") as f:
        model_columns = pickle.load(f)

    with open(CAP_LIMITS_PATH, "rb") as f:
        cap_limits = pickle.load(f)

    with open(CATEGORY_VALUES_PATH, "rb") as f:
        category_values = pickle.load(f)

    with open(INPUT_RANGES_PATH, "rb") as f:
        input_ranges = pickle.load(f)

    return model, model_columns, cap_limits, category_values, input_ranges


def validate_inputs(raw_input: pd.DataFrame, input_ranges: dict) -> list:
    warnings = []

    for col in raw_input.columns:
        if col in input_ranges:
            value = float(raw_input.iloc[0][col])
            q1 = float(input_ranges[col]["q1"])
            q3 = float(input_ranges[col]["q3"])

            iqr = q3 - q1
            soft_lower = q1 - 1.5 * iqr
            soft_upper = q3 + 1.5 * iqr

            if value < soft_lower or value > soft_upper:
                warnings.append(
                    f"{col.replace('_', ' ').title()} is unusual compared to the training data."
                )

    return warnings


def enforce_input_bounds(raw_input: pd.DataFrame, input_ranges: dict) -> pd.DataFrame:
    df = raw_input.copy()

    for col in df.columns:
        if col in input_ranges:
            min_val = float(input_ranges[col]["min"])
            max_val = float(input_ranges[col]["max"])
            df[col] = df[col].clip(lower=min_val, upper=max_val)

    return df


def preprocess_input(raw_input_df: pd.DataFrame, model_columns: list, cap_limits: dict) -> pd.DataFrame:
    df = raw_input_df.copy()

    # --------------------------------
    # Outlier capping
    # --------------------------------
    for col, limits in cap_limits.items():
        if col in df.columns:
            lower = float(limits["lower"])
            upper = float(limits["upper"])
            df[col] = df[col].clip(lower=lower, upper=upper)

    # --------------------------------
    # Feature engineering
    # --------------------------------
    if {"temperature_motor", "vibration_rms"}.issubset(df.columns):
        df["temp_vibration"] = df["temperature_motor"] * df["vibration_rms"]

    if {"rpm", "pressure_level"}.issubset(df.columns):
        df["rpm_pressure_ratio"] = df["rpm"] / (df["pressure_level"] + 1)

    if {"temperature_motor", "ambient_temp"}.issubset(df.columns):
        df["temp_ambient_diff"] = df["temperature_motor"] - df["ambient_temp"]

    if {"hours_since_maintenance", "temperature_motor"}.issubset(df.columns):
        df["degradation_index"] = df["hours_since_maintenance"] * df["temperature_motor"]

    # --------------------------------
    # One-hot encoding
    # --------------------------------
    df = pd.get_dummies(df)

    # --------------------------------
    # Align with training columns
    # --------------------------------
    df = df.reindex(columns=model_columns, fill_value=0)

    return df


def get_feature_importance_df(model, model_columns, top_n=12) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])

    fi_df = pd.DataFrame({
        "Feature": model_columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return fi_df.head(top_n)


def create_gauge_chart(prediction: float, max_rul: float = 120) -> go.Figure:
    bounded_value = min(max(prediction, 0), max_rul)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bounded_value,
        number={"suffix": " hrs"},
        title={"text": "Predicted Remaining Useful Life"},
        gauge={
            "axis": {"range": [0, max_rul]},
            "bar": {"thickness": 0.28},
            "steps": [
                {"range": [0, max_rul * 0.25], "color": "#7f1d1d"},
                {"range": [max_rul * 0.25, max_rul * 0.50], "color": "#b45309"},
                {"range": [max_rul * 0.50, max_rul * 0.75], "color": "#065f46"},
                {"range": [max_rul * 0.75, max_rul], "color": "#064e3b"}
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.75,
                "value": bounded_value
            }
        }
    ))

    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


def create_sensor_profile_chart(input_df: pd.DataFrame, cap_limits: dict) -> go.Figure:
    sensor_cols = [
        "vibration_rms",
        "temperature_motor",
        "current_phase_avg",
        "pressure_level",
        "rpm",
        "hours_since_maintenance",
        "ambient_temp"
    ]

    rows = []
    for col in sensor_cols:
        if col in input_df.columns and col in cap_limits:
            upper = float(cap_limits[col]["upper"])
            value = float(input_df.iloc[0][col])
            relative_value = value / upper if upper not in [0, None] else 0.0

            rows.append({
                "Feature": col.replace("_", " ").title(),
                "Relative Level": relative_value
            })

    profile_df = pd.DataFrame(rows)

    if profile_df.empty:
        return go.Figure()

    fig = px.bar(
        profile_df,
        x="Feature",
        y="Relative Level",
        title="Sensor Profile (relative to capped upper limits)"
    )

    fig.update_layout(
        height=360,
        xaxis_title="",
        yaxis_title="Relative Level"
    )

    return fig


def create_feature_importance_chart(fi_df: pd.DataFrame) -> go.Figure:
    if fi_df.empty:
        return go.Figure()

    chart_df = fi_df.copy()
    chart_df["Feature"] = chart_df["Feature"].str.replace("_", " ").str.title()

    fig = px.bar(
        chart_df.sort_values("Importance", ascending=True),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Feature Importances"
    )

    fig.update_layout(
        height=420,
        xaxis_title="Importance",
        yaxis_title=""
    )

    return fig


def rul_status(prediction: float):
    if prediction < 20:
        return "Critical", "Schedule maintenance immediately.", "status-critical"
    elif prediction < 50:
        return "Warning", "Monitor the machine closely and plan maintenance soon.", "status-warn"
    elif prediction < 80:
        return "Stable", "Machine condition appears manageable.", "status-good"
    return "Healthy", "Machine appears to have comfortable remaining life.", "status-good"


# =========================================================
# Load Artifacts
# =========================================================
try:
    model, model_columns, cap_limits, category_values, input_ranges = load_artifacts()
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# =========================================================
# Title
# =========================================================
st.title("🛠️ Predictive Maintenance - Advanced RUL Dashboard")
st.markdown(
    '<div class="small-note">Interactive dashboard for estimating Remaining Useful Life (RUL) using the tuned Random Forest model.</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# =========================================================
# Sidebar Inputs
# =========================================================
st.sidebar.header("Machine Inputs")

machine_type_options = category_values.get("machine_type", [])
operating_mode_options = category_values.get("operating_mode", [])

if not machine_type_options or not operating_mode_options:
    st.error("Category values are missing or invalid in category_values.pkl.")
    st.stop()

machine_type = st.sidebar.selectbox("Machine Type", machine_type_options)
operating_mode = st.sidebar.selectbox("Operating Mode", operating_mode_options)

st.sidebar.markdown("### Sensor Readings")

vibration_rms = st.sidebar.number_input(
    "Vibration RMS",
    min_value=float(input_ranges["vibration_rms"]["min"]),
    max_value=float(input_ranges["vibration_rms"]["max"]),
    value=float(input_ranges["vibration_rms"]["median"]),
    step=0.1
)

temperature_motor = st.sidebar.number_input(
    "Motor Temperature",
    min_value=float(input_ranges["temperature_motor"]["min"]),
    max_value=float(input_ranges["temperature_motor"]["max"]),
    value=float(input_ranges["temperature_motor"]["median"]),
    step=0.1
)

current_phase_avg = st.sidebar.number_input(
    "Current Phase Average",
    min_value=float(input_ranges["current_phase_avg"]["min"]),
    max_value=float(input_ranges["current_phase_avg"]["max"]),
    value=float(input_ranges["current_phase_avg"]["median"]),
    step=0.1
)

pressure_level = st.sidebar.number_input(
    "Pressure Level",
    min_value=float(input_ranges["pressure_level"]["min"]),
    max_value=float(input_ranges["pressure_level"]["max"]),
    value=float(input_ranges["pressure_level"]["median"]),
    step=0.1
)

rpm = st.sidebar.number_input(
    "RPM",
    min_value=float(input_ranges["rpm"]["min"]),
    max_value=float(input_ranges["rpm"]["max"]),
    value=float(input_ranges["rpm"]["median"]),
    step=1.0
)

hours_since_maintenance = st.sidebar.number_input(
    "Hours Since Maintenance",
    min_value=float(input_ranges["hours_since_maintenance"]["min"]),
    max_value=float(input_ranges["hours_since_maintenance"]["max"]),
    value=float(input_ranges["hours_since_maintenance"]["median"]),
    step=1.0
)

ambient_temp = st.sidebar.number_input(
    "Ambient Temperature",
    min_value=float(input_ranges["ambient_temp"]["min"]),
    max_value=float(input_ranges["ambient_temp"]["max"]),
    value=float(input_ranges["ambient_temp"]["median"]),
    step=0.1
)

predict_btn = st.sidebar.button("Predict RUL", use_container_width=True)

# =========================================================
# Raw Input
# =========================================================
raw_input = pd.DataFrame([{
    "machine_type": machine_type,
    "operating_mode": operating_mode,
    "vibration_rms": vibration_rms,
    "temperature_motor": temperature_motor,
    "current_phase_avg": current_phase_avg,
    "pressure_level": pressure_level,
    "rpm": rpm,
    "hours_since_maintenance": hours_since_maintenance,
    "ambient_temp": ambient_temp
}])

# =========================================================
# Overview Section
# =========================================================
left_col, right_col = st.columns([1.15, 1.0])

with left_col:
    st.markdown('<div class="section-title">Input Overview</div>', unsafe_allow_html=True)
    st.dataframe(raw_input, use_container_width=True)

with right_col:
    st.markdown('<div class="section-title">Model Information</div>', unsafe_allow_html=True)
    st.info(
        "This app uses the tuned Random Forest model trained on cleaned, "
        "capped, encoded, and feature-engineered machine data."
    )

# =========================================================
# Prediction
# =========================================================
if predict_btn:
    try:
        warnings = validate_inputs(raw_input, input_ranges)

        if warnings:
            st.warning("⚠️ Please review these unusual inputs:\n\n- " + "\n- ".join(warnings))

        safe_input = enforce_input_bounds(raw_input, input_ranges)
        processed_input = preprocess_input(safe_input, model_columns, cap_limits)

        prediction = float(model.predict(processed_input)[0])
        prediction = max(prediction, 0.0)

        status_label, status_note, status_class = rul_status(prediction)

        # -----------------------------------------
        # Metrics
        # -----------------------------------------
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Predicted RUL</div>
                <div class="metric-value">{prediction:.2f} hrs</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Condition Status</div>
                <div class="metric-value {status_class}">{status_label}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            maintenance_priority = max(0, min(100, 100 - prediction))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Maintenance Priority</div>
                <div class="metric-value">{maintenance_priority:.0f}/100</div>
            </div>
            """, unsafe_allow_html=True)

        st.success(status_note)

        # -----------------------------------------
        # Main Visuals
        # -----------------------------------------
        g1, g2 = st.columns([1.1, 1.1])

        with g1:
            st.plotly_chart(
                create_gauge_chart(prediction, max_rul=120),
                use_container_width=True
            )

        with g2:
            st.plotly_chart(
                create_sensor_profile_chart(safe_input, cap_limits),
                use_container_width=True
            )

        # -----------------------------------------
        # Feature importance + processed preview
        # -----------------------------------------
        fi_df = get_feature_importance_df(model, model_columns, top_n=12)

        b1, b2 = st.columns([1.2, 1.0])

        with b1:
            st.plotly_chart(
                create_feature_importance_chart(fi_df),
                use_container_width=True
            )

        with b2:
            st.markdown('<div class="section-title">Processed Input Preview</div>', unsafe_allow_html=True)
            non_zero = processed_input.loc[:, (processed_input != 0).any(axis=0)]
            st.dataframe(non_zero, use_container_width=True)

        # -----------------------------------------
        # Interpretation
        # -----------------------------------------
        st.markdown('<div class="section-title">Interpretation</div>', unsafe_allow_html=True)
        st.write(
            "The prediction is generated from machine type, operating mode, sensor readings, "
            "maintenance history, capped numerical values, and engineered features such as "
            "temperature-vibration interaction and degradation index. The feature importance "
            "chart reflects the model’s global behavior, not a case-specific local explanation."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

else:
    st.markdown('<div class="section-title">How to Use</div>', unsafe_allow_html=True)
    st.write(
        "Select the machine type and operating mode from the sidebar, enter the sensor values, "
        "and click **Predict RUL**. The dashboard will display the predicted remaining useful life, "
        "a gauge-based visual summary, sensor profile, and top feature importances."
    )