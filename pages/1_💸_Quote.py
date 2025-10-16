# pages/1_💬_Quote.py
import os
import sys
import io
import json
import traceback
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------
# Ensure src/ is importable
# ---------------------------------
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

st.set_page_config(page_title="💬 Quote", layout="wide")
st.title("💬 Quote")

# ---------------------------------
# Import model
# ---------------------------------
try:
    import src.model as model
except Exception:
    st.error("❌ Import of src.model failed:")
    st.code(traceback.format_exc())
    st.stop()

# ---------------------------------
# Helper: load version.json
# ---------------------------------
def load_version_from_io(f) -> dict:
    try:
        return json.load(f)
    except Exception:
        f.seek(0)
        return json.loads(f.read().decode("utf-8"))

def ensure_version():
    # 1) Prefer the version built in Admin this session
    v = st.session_state.get("version")
    if v:
        return v, "session (Admin)"

    # 2) Optional: load a default version.json from repo (if you keep one)
    default_path = os.environ.get("ULP_VERSION_PATH", os.path.join(APP_ROOT, "version.json"))
    if os.path.exists(default_path):
        try:
            with open(default_path, "r") as f:
                return json.load(f), f"file: {default_path}"
        except Exception:
            pass

    # 3) Fall back to user upload
    return None, None

version, source = ensure_version()

with st.expander("⚙️ Version Source", expanded=True):
    if version:
        st.success(f"Using version from {source}")
    else:
        st.warning("No version in session or default file. Upload a version.json below.")
    up = st.file_uploader("Upload version.json (from Admin → Download)", type=["json"])
    if up is not None:
        try:
            version = load_version_from_io(up)
            source = f"upload: {getattr(up, 'name', 'version.json')}"
            st.success(f"Loaded version from {source}")
        except Exception:
            st.error("Could not parse uploaded version.json")
            st.stop()

if not version:
    st.info("🧩 Build a version in the Admin page or upload a version.json here to quote.")
    st.stop()

# ---------------------------------
# Guard info (if present)
# ---------------------------------
guard = version.get("guard")
if guard and guard.get("low") is not None and guard.get("high") is not None:
    st.caption(f"Guard active: margins will be clipped to {guard['low']*100:.0f}%–{guard['high']*100:.0f}% where cost is provided.")
else:
    st.caption("Guard not set in version. (You can enable it in Admin when building a version.)")

# ---------------------------------
# Single Quote
# ---------------------------------
st.subheader("Single Quote")

c1, c2, c3 = st.columns([1,1,1])
msrp = c1.number_input("MSRP", min_value=0.01, value=1000.00, step=1.0, format="%.2f")
state = c2.text_input("Destination State (2-letter)", value="TX").strip().upper()
cost  = c3.number_input("Cost to ULP (optional but recommended to enforce guard)", min_value=0.00, value=0.00, step=1.0, format="%.2f")

def _compute_single(msrp: float, state: str, cost: float) -> dict:
    # If cost is >0, apply guard precisely
    if cost and cost > 0:
        price = model.price_for_with_cost(msrp, cost, state, version)
        margin = (price - cost) / cost
    else:
        price = model.price_for(msrp, state, version)
        margin = np.nan  # unknown without cost
    return {"price": float(price), "margin": float(margin) if not np.isnan(margin) else None}

if st.button("Quote"):
    try:
        out = _compute_single(msrp, state, cost)
        colA, colB = st.columns(2)
        colA.metric("Price", f"${out['price']:,.2f}")
        if out["margin"] is not None:
            colB.metric("Margin", f"{out['margin']*100:.2f}%")
        else:
            colB.caption("Margin requires Cost to ULP.")
    except Exception:
        st.error("Quote failed:")
        st.code(traceback.format_exc())

st.divider()

# ---------------------------------
# Bulk Quote (CSV/XLSX)
# ---------------------------------
st.subheader("Bulk Quote (CSV or Excel)")

st.caption("Expected columns (auto-detected, case-insensitive): **MSRP**, **Cost to ULP**, **Destination State**")
bulk_file = st.file_uploader("Upload file", type=["csv", "xlsx"], key="bulk")

def _read_bulk(f):
    name = (getattr(f, "name", "") or "").lower()
    if name.endswith(".csv"):
        df = pd.read_csv(f)
    else:
        df = pd.read_excel(f)
    # Map columns to our names (reuse model loader’s heuristic)
    cols = list(df.columns)
    msrp_col  = model._pick_column(cols, "msrp", 0)
    cost_col  = model._pick_column(cols, "cost to ulp", 1 if len(cols)>1 else 0)
    state_col = model._pick_column(cols, "destination state", 2 if len(cols)>2 else 0)
    df = df.rename(columns={msrp_col:"MSRP", cost_col:"Cost_to_ULP", state_col:"Destination_State"}).copy()
    df["MSRP"] = pd.to_numeric(df["MSRP"], errors="coerce")
    df["Cost_to_ULP"] = pd.to_numeric(df["Cost_to_ULP"], errors="coerce")
    df["Destination_State"] = df["Destination_State"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["MSRP","Destination_State"])
    return df

def _quote_bulk(df: pd.DataFrame) -> pd.DataFrame:
    # Choose with/without cost path
    has_cost = "Cost_to_ULP" in df.columns and df["Cost_to_ULP"].notna().any()
    if has_cost:
        prices = [
            model.price_for_with_cost(msrp, float(cost) if cost==cost else 0.0, state, version)
            for msrp, cost, state in zip(df["MSRP"].values, df["Cost_to_ULP"].fillna(0).values, df["Destination_State"].values)
        ]
        margins = np.where(df["Cost_to_ULP"].fillna(0).values > 0,
                           (np.array(prices) - df["Cost_to_ULP"].fillna(0).values) / np.where(df["Cost_to_ULP"].fillna(0).values>0, df["Cost_to_ULP"].fillna(0).values, 1.0),
                           np.nan)
        out = df.copy()
        out["Price"] = np.round(prices, 2)
        out["Margin%"] = np.where(np.isnan(margins), np.nan, np.round(margins*100.0, 2))
        return out
    else:
        prices = [model.price_for(msrp, state, version) for msrp, state in zip(df["MSRP"].values, df["Destination_State"].values)]
        out = df.copy()
        out["Price"] = np.round(prices, 2)
        out["Margin%"] = np.nan
        return out

if bulk_file is not None:
    try:
        bdf = _read_bulk(bulk_file)
        st.write(f"Detected {len(bdf):,} rows")
        quoted = _quote_bulk(bdf)
        st.dataframe(quoted.head(50))
        # Download
        buf = io.BytesIO()
        if bulk_file.name.lower().endswith(".csv"):
            quoted.to_csv(buf, index=False)
            mime = "text/csv"; fname = "quotes.csv"
        else:
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                quoted.to_excel(writer, index=False, sheet_name="Quotes")
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            fname = "quotes.xlsx"
        st.download_button("💾 Download quoted file", data=buf.getvalue(), file_name=fname, mime=mime)
    except Exception:
        st.error("Bulk quoting failed:")
        st.code(traceback.format_exc())

# ---------------------------------
# Footer / debug
# ---------------------------------
with st.expander("🔎 Debug"):
    st.json({"version_source": source, "guard": guard})
