# model_page.py
# Streamlit "Model" page with zone-bias adjustments, caps, and analytics
# Assumes input data has columns:
#   ["MSRP", "Cost to ULP", "Price Model", "Destination State"]

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------
ZONE_MAP = {
    'CA':'West','OR':'West','WA':'West','NV':'West','AZ':'West','CO':'West','UT':'West','NM':'West','ID':'West','MT':'West','WY':'West',
    'TX':'South','FL':'South','GA':'South','AL':'South','LA':'South','NC':'South','SC':'South','TN':'South','MS':'South','AR':'South','OK':'South','KY':'South','WV':'South',
    'OH':'Midwest','MI':'Midwest','IL':'Midwest','WI':'Midwest','IN':'Midwest','MN':'Midwest','IA':'Midwest','MO':'Midwest','KS':'Midwest','NE':'Midwest','SD':'Midwest','ND':'Midwest',
    'PA':'East','NJ':'East','NY':'East','MD':'East','VA':'East','MA':'East','CT':'East','DE':'East','RI':'East','NH':'East','VT':'East','ME':'East','DC':'East'
}

DEFAULT_ZONE_MULTS = {
    "West": 1.10,     # +10%
    "East": 1.05,     # +5%
    "Midwest": 0.95,  # -5%
    "South": 1.00,    # baseline
    "Other": 1.08     # +8% (AK/HI/PR and anything unmapped)
}

BAND_BINS = [-np.inf, 10, 40, np.inf]
BAND_LABELS = ["Below 10%", "Inside 10–40%", "Above 40%"]

def infer_zone(state: str) -> str:
    if pd.isna(state):
        return "Other"
    state = str(state).strip().upper()
    return ZONE_MAP.get(state, "Other")

def apply_caps(series: pd.Series, floor: float, ceiling: float) -> pd.Series:
    return series.clip(lower=floor, upper=ceiling)

def compute_metrics(df: pd.DataFrame, price_col: str, cost_col: str = "Cost to ULP") -> dict:
    out = {}
    # margin as % of revenue
    margin = (df[price_col] - df[cost_col]) / df[price_col] * 100
    out["mean_margin"] = float(np.nanmean(margin))
    out["std_margin"]  = float(np.nanstd(margin))
    out["mae"]         = float(np.nanmean((df[price_col] - df[cost_col]).abs()))
    out["mpe"]         = float(np.nanmean((df[price_col] - df[cost_col]) / df[cost_col]) * 100)
    out["corr"]        = float(df[[cost_col, price_col]].corr().iloc[0,1])
    bands = pd.cut(margin, bins=BAND_BINS, labels=BAND_LABELS)
    band_counts = bands.value_counts(normalize=True).reindex(BAND_LABELS, fill_value=0) * 100
    out["band_counts"] = band_counts
    out["margin_series"] = margin
    return out

def add_cost_tiers(df: pd.DataFrame, cost_col: str = "Cost to ULP") -> pd.Series:
    bins = [0, 2000, 5000, 10000, np.inf]
    labels = ['<$2k', '$2k–5k', '$5k–10k', '$10k+']
    return pd.cut(df[cost_col], bins=bins, labels=labels)

# -----------------------------
# Single-quote function (importable from your Quote page)
# -----------------------------
def price_single_quote(cost_to_ulp: float, dest_state: str,
                       base_model_price: float,
                       min_charge: float = 300.0,
                       max_charge: float = 8000.0,
                       zone_mults: dict = None) -> float:
    """
    Returns adjusted price for one quote using:
      - cap/floor: min_charge, max_charge
      - zone bias correction via zone_mults
    """
    if zone_mults is None:
        zone_mults = DEFAULT_ZONE_MULTS

    # Base cap/floor
    p = np.clip(base_model_price, min_charge, max_charge)

    # Zone adjustment
    zone = infer_zone(dest_state)
    mult = float(zone_mults.get(zone, 1.08))
    p = p * mult

    # Re-apply caps after multiplier to avoid escaping the band
    p = float(np.clip(p, min_charge, max_charge))
    return p

# -----------------------------
# Streamlit UI
# -----------------------------
def run():
    st.title("Pricing Model — Zone-Adjusted with Caps")

    st.markdown(
        "Upload your dataset with **Cost to ULP**, **Price Model**, and **Destination State**. "
        "This page will apply a $300 floor and $8,000 cap (editable), correct for **Zone bias**, "
        "and show performance statistics, distributions, and exports."
    )

    # Controls
    with st.sidebar:
        st.header("Controls")
        min_charge = st.number_input("Minimum Charge ($)", min_value=0.0, value=300.0, step=25.0)
        max_charge = st.number_input("Maximum Charge ($)", min_value=0.0, value=8000.0, step=100.0)

        st.markdown("**Zone Multipliers** (edit to re-calibrate):")
        z_west = st.number_input("West", value=DEFAULT_ZONE_MULTS["West"], step=0.01, format="%.2f")
        z_east = st.number_input("East", value=DEFAULT_ZONE_MULTS["East"], step=0.01, format="%.2f")
        z_mid  = st.number_input("Midwest", value=DEFAULT_ZONE_MULTS["Midwest"], step=0.01, format="%.2f")
        z_south= st.number_input("South", value=DEFAULT_ZONE_MULTS["South"], step=0.01, format="%.2f")
        z_other= st.number_input("Other", value=DEFAULT_ZONE_MULTS["Other"], step=0.01, format="%.2f")

        zone_mults = {
            "West": float(z_west),
            "East": float(z_east),
            "Midwest": float(z_mid),
            "South": float(z_south),
            "Other": float(z_other)
        }

        st.divider()
        st.caption("Tip: Use the same multipliers in your Quote page by importing price_single_quote().")

    uploaded = st.file_uploader("Upload .xlsx / .csv", type=["xlsx","csv"])

    if uploaded is None:
        st.info("Waiting for file…")
        return

    # Load data
    if uploaded.name.lower().endswith(".xlsx"):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded)

    # Basic validation
    required = {"Cost to ULP","Price Model","Destination State"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # Ensure numeric columns
    df["Cost to ULP"] = pd.to_numeric(df["Cost to ULP"], errors="coerce")
    df["Price Model"] = pd.to_numeric(df["Price Model"], errors="coerce")

    # Drop obvious bad rows
    df = df.dropna(subset=["Cost to ULP","Price Model"]).copy()

    # Add Zone
    df["Zone"] = df["Destination State"].apply(infer_zone)

    # Apply base caps/floor first
    df["Capped Price"] = apply_caps(df["Price Model"], min_charge, max_charge)

    # Zone bias correction
    df["Zone Mult"] = df["Zone"].map(zone_mults).fillna(zone_mults["Other"]).astype(float)
    df["Adjusted Price"] = df["Capped Price"] * df["Zone Mult"]

    # Re-cap after multiplier to enforce final envelope
    df["Adjusted Price"] = apply_caps(df["Adjusted Price"], min_charge, max_charge)

    # Metrics before/after
    base_metrics  = compute_metrics(df.assign(ModelPrice=df["Price Model"]), "ModelPrice")
    capped_metrics = compute_metrics(df.assign(ModelPrice=df["Capped Price"]), "ModelPrice")
    final_metrics = compute_metrics(df.assign(ModelPrice=df["Adjusted Price"]), "ModelPrice")

    st.subheader("Summary Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Unadjusted (raw model)**")
        st.write(f"Mean Margin: {base_metrics['mean_margin']:.2f}%")
        st.write(f"MAE: ${base_metrics['mae']:.2f}")
        st.write(f"MPE: {base_metrics['mpe']:.2f}%")
        st.write(f"Corr (Cost↔Price): {base_metrics['corr']:.3f}")
    with col2:
        st.markdown("**Capped (min/max only)**")
        st.write(f"Mean Margin: {capped_metrics['mean_margin']:.2f}%")
        st.write(f"MAE: ${capped_metrics['mae']:.2f}")
        st.write(f"MPE: {capped_metrics['mpe']:.2f}%")
        st.write(f"Corr (Cost↔Price): {capped_metrics['corr']:.3f}")
    with col3:
        st.markdown("**Final (cap + zone bias)**")
        st.write(f"**Mean Margin:** **{final_metrics['mean_margin']:.2f}%**")
        st.write(f"**MAE:** **${final_metrics['mae']:.2f}**")
        st.write(f"**MPE:** **{final_metrics['mpe']:.2f}%**")
        st.write(f"**Corr (Cost↔Price):** **{final_metrics['corr']:.3f}**")

    st.markdown("**Margin Distribution (% of quotes in band)**")
    dist = pd.DataFrame({
        "Raw": base_metrics["band_counts"].round(2),
        "Capped": capped_metrics["band_counts"].round(2),
        "Final": final_metrics["band_counts"].round(2)
    })
    st.dataframe(dist)

    # Cost tiers and zones
    st.subheader("Structural Performance")
    work = df.copy()
    work["Adjusted Margin %"] = (work["Adjusted Price"] - work["Cost to ULP"]) / work["Adjusted Price"] * 100
    work["Cost Tier"] = add_cost_tiers(work)

    t1 = work.groupby("Cost Tier")["Adjusted Margin %"].agg(["mean","std","count"]).round(2)
    t2 = work.groupby("Zone")["Adjusted Margin %"].agg(["mean","std","count"]).round(2)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**By Cost Tier (Final)**")
        st.dataframe(t1)
    with c2:
        st.markdown("**By Zone (Final)**")
        st.dataframe(t2)

    # Visuals
    st.subheader("Visuals")
    # Scatter: Cost to ULP vs Adjusted Price
    fig1, ax1 = plt.subplots(figsize=(7,5))
    ax1.scatter(work["Cost to ULP"], work["Adjusted Price"], alpha=0.5)
    ax1.set_xlabel("Cost to ULP")
    ax1.set_ylabel("Adjusted Price")
    ax1.set_title("Adjusted Price vs Cost (with caps & zone correction)")
    ax1.grid(True)
    st.pyplot(fig1)

    # Histogram: Adjusted Margin
    fig2, ax2 = plt.subplots(figsize=(7,4))
    ax2.hist(work["Adjusted Margin %"], bins=40, alpha=0.8)
    ax2.set_title("Adjusted Margin Distribution")
    ax2.set_xlabel("Margin (%)")
    ax2.set_ylabel("Count")
    ax2.grid(True)
    st.pyplot(fig2)

    # Export
    st.subheader("Export Adjusted Quotes")
    export_cols = ["MSRP","Cost to ULP","Destination State","Zone","Price Model","Capped Price","Zone Mult","Adjusted Price","Adjusted Margin %","Cost Tier"]
    export_df = work[export_cols].copy()

    csv_buf = io.StringIO()
    export_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download CSV (Final Adjusted)",
        data=csv_buf.getvalue(),
        file_name="adjusted_quotes_zone_capped.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    run()
