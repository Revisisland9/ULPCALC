import sys, pkgutil, platform
import streamlit as st

st.title("Environment Debug")
st.write("Python:", sys.version)
st.write("Platform:", platform.platform())
st.write("Sys.path:", sys.path)

mods = sorted([m.name for m in pkgutil.iter_modules()])
st.write("Has plotly?", "plotly" in mods)
st.write("First 50 modules:", mods[:50])

try:
    import plotly, plotly.express as px
    st.success(f"Plotly version: {plotly.__version__}")
except Exception as e:
    st.error(f"Plotly import failed: {e}")
