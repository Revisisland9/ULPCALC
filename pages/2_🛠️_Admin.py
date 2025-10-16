# pages/2_🛠️_Admin.py
import os, sys
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)
import io
import pandas as pd
import streamlit as st
from src.model import load_dataframe, CalibParams, build_version
from src.storage import save_upload, publish_version, list_versions, load_version
from src.ui import zone_table_component, metrics_cards

st.set_page_config(page_title="Admin", page_icon="🛠️", layout="wide")
st.title("Admin — Upload & Calibrate")

# ----------------------------
# 1) Upload + Column Mapping
# ----------------------------
with st.expander("1) Upload new actuals (MSRP, Cost to ULP, Destination State)"):
    up = st.file_uploader("CSV or XLSX", type=["csv","xlsx"])
    uploaded_path = None
    msrp_col = cost_col = state_col = None

    if up is not None:
        suffix = ".csv" if up.name.lower().endswith(".csv") else ".xlsx"
        uploaded_path = save_upload(up, suffix)
        st.success(f"Saved to {uploaded_path}")

        # Show columns and let the user map them explicitly
        src_preview = pd.read_csv(uploaded_path) if suffix==".csv" else pd.read_excel(uploaded_path)
        st.caption("Detected columns:")
        st.write(list(src_preview.columns))

        c1, c2, c3 = st.columns(3)
        msrp_col  = c1.selectbox("Which column is MSRP?", list(src_preview.columns))
        cost_col  = c2.selectbox("Which column is Cost to ULP?", list(src_preview.columns))
        state_col = c3.selectbox("Which column is Destination State?", list(src_preview.columns))

        # Tiny preview of the selected columns
        prev_cols = src_preview[[msrp_col, cost_col, state_col]].head(10).copy()
        prev_cols.columns = ["MSRP", "Cost to ULP", "Destination State"]
        st.dataframe(prev_cols, use_container_width=True)

st.markdown("---")

# ----------------------------
# 2) Calibration Parameters
# ----------------------------
st.subheader("2) Calibration Parameters")
c1, c2, c3, c4 = st.columns(4)
target_mean = c1.number_input("Target Mean Margin", value=0.18, min_value=0.05, max_value=0.50, step=0.01, format="%.2f")
band_low    = c2.number_input("Band Low", value=0.10, min_value=0.00, max_value=0.40, step=0.01, format="%.2f")
band_high   = c3.number_input("Band High", value=0.40, min_value=0.10, max_value=0.60, step=0.01, format="%.2f")
band_target = c4.number_input("Coverage Goal (%% inside band)", value=0.95, min_value=0.80, max_value=0.99, step=0.01, format="%.2f")

params = CalibParams(
    target_mean=target_mean,
    band_low=band_low,
    band_high=band_high,
    band_target=band_target,
    # Temporarily disable the 5% change cap so we can fully re-center
    change_cap_pct=1.00
)

st.markdown("---")

# ----------------------------
# 3) Recalibrate (Preview)
# ----------------------------
st.subheader("3) Recalibrate (Preview)")
colA, colB = st.columns([1,2])
with colA:
    run_btn = st.button("Recalibrate with latest upload", type="primary", disabled=(uploaded_path is None or msrp_col is None))

with colB:
    st.info("Recalibration learns zones & multipliers from your uploaded actuals. It won’t go live until you Publish.")

if run_btn and uploaded_path:
    # Load the dataframe using the chosen columns (no heuristics)
    df = load_dataframe(uploaded_path, msrp_col=msrp_col, cost_col=cost_col, state_col=state_col)

    # Quick sanity stats on raw data
    st.caption(f"Rows after cleaning: {len(df):,}")
    if len(df):
        median_ratio = float((df['Cost_to_ULP']/df['MSRP']).median())
        st.caption(f"Median Cost/MSRP in upload: {median_ratio:.3f} (if this is ~0.90, many lanes are inherently tight)")

    # Build version (zones/ratios), calibrate, normalize, score
    prev = load_version()
    version, state_map, zone_table = build_version(df, params, prev)

    # KPI cards from the model
    st.success("Calibration complete (preview). Review below.")
    metrics_cards(version["metrics"])
    zone_table_component(zone_table)

    # ---------------------------------------
    # 3b) HARD SANITY: recompute live revenue
    # ---------------------------------------
    st.subheader("Sanity Check Against Exported Tables")
    zones_df = pd.DataFrame(version["zones"]).rename(
        columns={"zone":"Zone","expected_cost_ratio":"Zone_Expected_Cost_Ratio","multiplier":"Zone_Multiplier"}
    )
    state_join = pd.DataFrame(version["state_map"]).rename(columns={"state":"Destination_State","zone":"Zone"})
    check = (df.merge(state_join, on="Destination_State", how="left")
               .merge(zones_df, on="Zone", how="left"))
    check["Predicted_Price"] = check["MSRP"] * check["Zone_Expected_Cost_Ratio"] * check["Zone_Multiplier"]
    # compute totals and mean
    total_cost = float(check["Cost_to_ULP"].sum())
    total_rev  = float(check["Predicted_Price"].sum())
    mean_margin = float(((check["Predicted_Price"] - check["Cost_to_ULP"]) / check["Cost_to_ULP"]).mean())

    c1, c2, c3 = st.columns(3)
    c1.metric("Σ Cost", f"${total_cost:,.0f}")
    c2.metric("Target Σ Revenue", f"${total_cost*(1+params.target_mean):,.0f}")
    c3.metric("Achieved Σ Revenue", f"${total_rev:,.0f}")

    st.metric("Mean Margin (recomputed from exported tables)", f"{mean_margin*100:.2f}%",
              help="This should be ~ equal to Target Mean. If not, we know exactly where to look.")

    # Guardrails suggestion (optional)
    ok_mean = abs(mean_margin - params.target_mean) <= 0.01
    ok_band = version["metrics"]["pct_inside"] >= params.band_target - 0.01
    if not (ok_mean and ok_band):
        st.warning("Guardrails not met (mean margin or band coverage). You can still publish for testing, but consider re-running or adjusting parameters.")

    # ----------------------------
    # 4) Publish
    # ----------------------------
    st.subheader("4) Publish")
    if st.button("Publish This Version"):
        path = publish_version(version)
        st.success(f"Published as {path}")

st.markdown("---")
st.subheader("5) Versions")
vers = list_versions()
if not vers:
    st.caption("No versions yet.")
else:
    vv = pd.DataFrame(vers)
    st.dataframe(vv, use_container_width=True)
