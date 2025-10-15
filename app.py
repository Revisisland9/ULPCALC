# app.py
import streamlit as st

# Page configuration (sets window title and layout)
st.set_page_config(
    page_title="Pricing Model",
    page_icon="💸",
    layout="wide"
)

# Simple landing message
st.title("💸 Prospect Logistics Pricing App")

st.write("""
Welcome to your interactive pricing model.

Use the sidebar on the left to:
- **💸 Quote:** Generate quotes for any destination state and MSRP.
- **🛠️ Admin:** Upload new shipment data, recalibrate the model, and publish new versions.

Each recalibration automatically learns the latest cost behavior across states,
keeps 95% of shipments in the 10–40% p
