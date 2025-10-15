# pages/1_💸_Quote.py
import streamlit as st
import pandas as pd
from src.storage import load_version
from src.ui import state_map_component

st.set_page_config(page_title="Quote", page_icon="💸", layout="centered")

st.title("Get a Quote")
ver = load_version()
if not ver:
    st.warning("No published pricing version yet. Please publish one from the Admin page.")
    st.stop()

state_map = pd.DataFrame(ver["state_map"])
zones = pd.DataFrame(ver["zones"])

# join to bring ratios+mults onto state rows
state_map = state_map.merge(
    zones.rename(columns={"zone":"Zone", "expected_cost_ratio":"Zone_Expected_Cost_Ratio", "multiplier":"Zone_Multiplier"}),
    left_on="zone", right_on="Zone", how="left"
).rename(columns={"state":"Destination_State"})

col1, col2 = st.columns([1,1])
with col1:
    state_code = st.text_input("Destination State (2 letters)", "MO").strip().upper()
with col2:
    msrp = st.number_input("MSRP ($)", min_value=0.0, value=1000.0, step=10.0, format="%.2f")

state_map_component(state_map, state_code, msrp)
st.caption(f"Model version: {ver['version_id']}")
