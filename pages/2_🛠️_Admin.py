# pages/2_🛠️_Admin.py
import os
import sys
import json
import traceback
import pandas as pd
import streamlit as st

# ---------------------------------
# Make sure project root is importable
# ---------------------------------
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

st.set_page_config(page_title="🛠️ Admin Calibration", layout="wide")
st.title("🛠️ Admin Calibration")

# ---------------------------------
# Import model module
# ---------------------------------
try:
    import src.model as model
except Exception:
    st.error("❌ Import of src.model failed:")
    st.code(traceback.format_exc())
    st.stop()

with st.expander("ℹ️ Diagnostics", expanded=False):
    st.caption(f"model file: {getattr(model, '__file__', 'unknown')}")
    st.write("symbols:", sorted([n for n in dir(model) if not n.startswith("_")]))

st.markdown("Upload data, tune parameters, and build a new version payload.")

# ---------------------------------
# File upload
# ---------------------------------
uploaded_file = st.file_uploader("Upload dataset (CSV or Excel)", type=["csv", "xlsx"])
prev_json_file = st.file_uploader("Optional: previous version.json (to cap changes)", type=["json"])

if not uploaded_file:
    st.info("⬆️ Upload a dataset to begin.")
    st.stop()

# Load data via model loader (handles file-like)
try:
    df = model.load_dataframe(uploaded_file)
    st.success(f"Loaded {len(df):,} rows")
    st.dataframe(df.head())
except Exception:
    st.error("Error loading file:")
    st.code(traceback.format_exc())
    st.stop()

# Optional previous version
prev_version = None
if prev_json_file is not None:
    try:
        prev_version = json.load(prev_json_file)
        st.caption("Previous version loaded (change caps will apply if enabled).")
    except Exception:
        st.warning("Could not parse previous JSON; ignoring.")

# ---------------------------------
# Sidebar parameters (new model aware)
# ---------------------------------
st.sidebar.header("Calibration Parameters")

# Profitability & band
target_mean = st.sidebar.number_input("Target Mean Margin", value=0.18, step=0.01, format="%.2f")
band_low    = st.sidebar.number_input("Band Low", value=0.10, step=0.01, format="%.2f")
band_high   = st.sidebar.number_input("Band High", value=0.40, step=0.01, format="%.2f")
band_target = st.sidebar.slider("Band Coverage Target", min_value=0.50, max_value=0.95, value=0.85, step=0.01)

# Zoning & tiers
zones       = st.sidebar.slider("Zones (state groupings)", 2, 8, 4)
use_fixed_breaks = st.sidebar.checkbox("Use fixed MSRP breaks instead of quantile tiers?", value=False)
if use_fixed_breaks:
    breaks_str = st.sidebar.text_input("Fixed breaks (comma-separated)", "0,500,1000,2000,inf")
    def _parse_breaks(s: str):
        parts = []
        for tok in s.split(","):
            tok = tok.strip().lower()
            if tok in ("inf", "+inf", "infinity"):
                parts.append(float("inf"))
            else:
                parts.append(float(tok))
        if parts[0] != 0.0:
            parts[0] = 0.0
        if parts[-1] != float("inf"):
            parts.append(float("inf"))
        return tuple(parts)
    try:
        msrp_breaks = _parse_breaks(breaks_str)
        tiers = len(msrp_breaks) - 1
        st.sidebar.caption(f"{tiers} fixed tiers")
    except Exception:
        msrp_breaks = (0.0, 500.0, 1000.0, 2000.0, float('inf'))
        st.sidebar.warning("Could not parse breaks; using default fixed breaks.")
else:
    msrp_breaks = None
    tiers = st.sidebar.slider("Quantile tiers (if not fixed)", 6, 24, 16, step=1)

# Optimizer
iters      = st.sidebar.slider("Iterations", 60, 400, 140, step=10)
lr_mean    = st.sidebar.slider("Learning rate: mean", 0.05, 0.60, 0.30, step=0.01)
lr_tail    = st.sidebar.slider("Learning rate: tails", 0.05, 0.40, 0.22, step=0.01)
b_cap      = st.sidebar.slider("Slope clamp |b|", 0.2, 2.0, 1.0, step=0.1)
shrinkage  = st.sidebar.slider("Zone shrinkage → global", 0.0, 1.0, 0.70, step=0.05)

# Per-state correction
with_c_state = st.sidebar.checkbox("Enable per-state correction (c_state)", value=True)
c_step   = st.sidebar.slider("c_state step scale", 0.01, 0.10, 0.03, step=0.01)
c_decay  = st.sidebar.slider("c_state decay (EMA)", 0.50, 0.99, 0.90, step=0.01)
c_cap    = st.sidebar.slider("c_state clamp", 0.02, 0.30, 0.12, step=0.01)

# Guard (runtime clip)
use_guard = st.sidebar.checkbox("Enable guard (clip margins at price time)", value=False)
if use_guard:
    guard_low  = st.sidebar.number_input("Guard Low", value=0.10, step=0.01, format="%.2f")
    guard_high = st.sidebar.number_input("Guard High", value=0.40, step=0.01, format="%.2f")
else:
    guard_low = None
    guard_high = None

# Stability / previous version caps (applied in model)
change_cap = st.sidebar.slider("Change cap % vs previous", 0.00, 0.20, 0.07, step=0.01)

# Build params
params = model.CalibParams(
    target_mean=target_mean,
    band_low=band_low,
    band_high=band_high,
    band_target=band_target,
    zones=zones,
    msrp_breaks=msrp_breaks,
    tiers=tiers,
    shrinkage=shrinkage,
    iters=iters,
    lr_mean=lr_mean,
    lr_tail=lr_tail,
    change_cap_pct=change_cap,
    b_cap=b_cap,
    with_c_state=with_c_state,
    c_state_step_scale=c_step,
    c_state_decay=c_decay,
    c_state_cap=c_cap,
    guard_low=guard_low,
    guard_high=guard_high,
)

# ---------------------------------
# Run calibration
# ---------------------------------
if st.button("Run Calibration"):
    with st.spinner("Calibrating…"):
        try:
            version, state_map, zone_table = model.build_version(df, params, prev=prev_version)
        except Exception:
            st.error("❌ Calibration failed:")
            st.code(traceback.format_exc())
            st.stop()

    st.success("✅ Calibration complete")

    # Metrics
    m_no  = version["metrics"]["no_guard"]
    m_yes = version["metrics"]["with_guard"]

    st.subheader("Results")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean Margin (no guard)", f"{m_no['mean_margin']*100:.2f}%")
    c2.metric("% Inside 10–40 (no guard)", f"{m_no['pct_inside']*100:.1f}%")
    c3.metric("% Below 10 (no guard)", f"{m_no['pct_below']*100:.1f}%")
    c4.metric("% Above 40 (no guard)", f"{m_no['pct_above']*100:.1f}%")
    c5.metric("% Loss (no guard)", f"{m_no['pct_loss']*100:.2f}%")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean Margin (with guard)", f"{m_yes['mean_margin']*100:.2f}%")
    c2.metric("% Inside 10–40 (with guard)", f"{m_yes['pct_inside']*100:.1f}%")
    c3.metric("% Below 10 (with guard)", f"{m_yes['pct_below']*100:.1f}%")
    c4.metric("% Above 40 (with guard)", f"{m_yes['pct_above']*100:.1f}%")
    c5.metric("% Loss (with guard)", f"{m_yes['pct_loss']*100:.2f}%")

    # Tables
    with st.expander("📊 Zone Table (stabilized ratio used in pricing)"):
        st.dataframe(zone_table)
    with st.expander("🗺️ State → Zone Map"):
        st.dataframe(state_map)
    with st.expander("🧮 Zone×Tier Multipliers (a,b)"):
        st.dataframe(pd.DataFrame(version["zt_multipliers"]))
    if version.get("state_adjust"):
        with st.expander("⚖️ Per-state corrections (c_state)"):
            st.dataframe(pd.DataFrame(version["state_adjust"]))
    with st.expander("📐 Tier normalization (mid, width)"):
        st.dataframe(pd.DataFrame(version["tier_norm"]))
    with st.expander("🏷️ Tiers"):
        st.json(version["tiers"])

    # Download payload
    st.download_button(
        "💾 Download Version JSON",
        data=json.dumps(version, indent=2),
        file_name="version.json",
        mime="application/json",
    )

else:
    st.info("Adjust parameters in the sidebar, then click **Run Calibration**.")
