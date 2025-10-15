# pages/2_🛠️_Admin.py
import streamlit as st
import pandas as pd
from src.model import load_dataframe, CalibParams, build_version
from src.storage import save_upload, publish_version, list_versions, load_version
from src.ui import zone_table_component, metrics_cards

st.set_page_config(page_title="Admin", page_icon="🛠️", layout="wide")
st.title("Admin — Upload & Calibrate")

with st.expander("1) Upload new actuals (MSRP, Cost_to_ULP, Destination_State)"):
    up = st.file_uploader("CSV or XLSX", type=["csv","xlsx"])
    uploaded_path = None
    if up is not None:
        suffix = ".csv" if up.name.lower().endswith(".csv") else ".xlsx"
        uploaded_path = save_upload(up, suffix)
        st.success(f"Saved to {uploaded_path}")

st.markdown("---")
st.subheader("2) Calibration Parameters")
c1, c2, c3, c4 = st.columns(4)
target_mean = c1.number_input("Target Mean Margin", value=0.18, min_value=0.05, max_value=0.40, step=0.01, format="%.2f")
band_low    = c2.number_input("Band Low", value=0.10, min_value=0.00, max_value=0.40, step=0.01, format="%.2f")
band_high   = c3.number_input("Band High", value=0.40, min_value=0.10, max_value=0.60, step=0.01, format="%.2f")
band_target = c4.number_input("Coverage Goal (%% inside band)", value=0.95, min_value=0.80, max_value=0.99, step=0.01, format="%.2f")

params = CalibParams(
    target_mean=target_mean,
    band_low=band_low,
    band_high=band_high,
    band_target=band_target
)

st.markdown("---")
st.subheader("3) Recalibrate (Preview)")
colA, colB = st.columns([1,2])
with colA:
    run_btn = st.button("Recalibrate with latest upload", type="primary", disabled=(uploaded_path is None))
with colB:
    st.info("Recalibration learns zones & multipliers from your uploaded actuals. It won’t go live until you Publish.")

if run_btn and uploaded_path:
    df = load_dataframe(uploaded_path)
    prev = load_version()
    version, state_map, zone_table = build_version(df, params, prev)

    st.success("Calibration complete (preview). Review below.")
    metrics_cards(version["metrics"])
    zone_table_component(zone_table)

    st.subheader("Publish")
    if st.button("Publish This Version"):
        path = publish_version(version)
        st.success(f"Published as {path}")

st.markdown("---")
st.subheader("4) Versions")
vers = list_versions()
if not vers:
    st.caption("No versions yet.")
else:
    vv = pd.DataFrame(vers)
    st.dataframe(vv, use_container_width=True)
