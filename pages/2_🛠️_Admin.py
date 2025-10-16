# pages/2_🛠️_Admin.py
import os
import sys
import streamlit as st
import pandas as pd
import json
import traceback

# ----------------------------
# Ensure project root and import model
# ----------------------------
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

try:
    from src.model import load_dataframe, CalibParams, build_version, price_for, price_for_many
except Exception as e:
    st.error("❌ Import failed in Admin page. Here’s the full error trace:")
    st.code(traceback.format_exc())
    st.stop()

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="🛠️ Admin Calibration", layout="wide")
st.title("🛠️ Admin Calibration Panel")

st.markdown(
    """
    Upload your cost/MSRP/state dataset, set calibration parameters, and run the model.  
    This page will build the new `(a,b)` multiplier version and show key metrics.
    """
)

# ----------------------------
# File uploader
# ----------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file:
    try:
        df = load_dataframe(uploaded_file)
        st.success(f"Loaded {len(df):,} rows")
        st.dataframe(df.head())
    except Exception as e:
        st.error("Error loading file:")
        st.code(traceback.format_exc())
        st.stop()
else:
    st.info("⬆️ Upload a CSV or Excel file to begin.")
    st.stop()

# ----------------------------
# Parameter sidebar
# ----------------------------
st.sidebar.header("Calibration Parameters")

target_mean = st.sidebar.number_input("Target Mean Margin", value=0.18, step=0.01)
band_low = st.sidebar.number_input("Band Low", value=0.10, step=0.01)
band_high = st.sidebar.number_input("Band High", value=0.40, step=0.01)
band_target = st.sidebar.number_input("Band Target Coverage", value=0.80, step=0.05)
zones = st.sidebar.slider("Zones (state groupings)", 2, 6, 4)
iters = st.sidebar.slider("Iterations", 50, 300, 120)
lr_mean = st.sidebar.slider("Learning Rate (Mean)", 0.05, 0.5, 0.30)
lr_tail = st.sidebar.slider("Learning Rate (Tail)", 0.05, 0.5, 0.15)
shrinkage = st.sidebar.slider("Zone Shrinkage", 0.0, 1.0, 0.8)
change_cap = st.sidebar.slider("Change Cap %", 0.0, 0.2, 0.07)

params = CalibParams(
    target_mean=target_mean,
    band_low=band_low,
    band_high=band_high,
    band_target=band_target,
    zones=zones,
    shrinkage=shrinkage,
    iters=iters,
    lr_mean=lr_mean,
    lr_tail=lr_tail,
    change_cap_pct=change_cap,
)

# ----------------------------
# Run calibration
# ----------------------------
if st.button("Run Calibration"):
    with st.spinner("Running calibration... this may take a minute"):
        try:
            version, state_map, zone_table = build_version(df, params)
        except Exception as e:
            st.error("Error during calibration:")
            st.code(traceback.format_exc())
            st.stop()

        st.success("✅ Calibration complete")

        # Display metrics
        m = version["metrics"]
        st.subheader("Calibration Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Margin", f"{m['mean_margin']*100:.2f}%")
        col2.metric("% Inside 10–40%", f"{m['pct_inside']*100:.1f}%")
        col3.metric("% Below 10%", f"{m['pct_below']*100:.1f}%")
        col4.metric("% Above 40%", f"{m['pct_above']*100:.1f}%")

        # Display tables
        with st.expander("📊 Zone Table"):
            st.dataframe(zone_table)

        with st.expander("🗺️ State → Zone Map"):
            st.dataframe(state_map)

        with st.expander("🧮 Zone×Tier Multipliers (a,b)"):
            st.dataframe(pd.DataFrame(version["zt_multipliers"]))

        with st.expander("📐 Tier Normalization"):
            st.dataframe(pd.DataFrame(version["tier_norm"]))

        # Save JSON for use in Quote/Bulk
        version_json = json.dumps(version, indent=2)
        st.download_button(
            "💾 Download Version JSON",
            version_json,
            file_name="version.json",
            mime="application/json",
        )

else:
    st.info("Adjust parameters in the sidebar, then click **Run Calibration**.")
