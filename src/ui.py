# src/ui.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def zone_table_component(zone_df: pd.DataFrame):
    st.subheader("Zone Coefficients")
    st.dataframe(zone_df, use_container_width=True)

def state_map_component(state_map: pd.DataFrame, state_code: str, msrp: float):
    df = state_map.copy()
    df["Predicted_Price"] = msrp * df["Zone_Expected_Cost_Ratio"] * df["Zone_Multiplier"]
    st.caption("Map shows predicted price for the entered MSRP in each state.")
    fig = px.choropleth(
        df,
        locations="Destination_State",
        locationmode="USA-states",
        color="Predicted_Price",
        hover_name="Destination_State",
        hover_data={
            "Predicted_Price":":.2f",
            "Destination_State":False,
            "Zone":True,
            "Zone_Expected_Cost_Ratio":":.3f",
            "Zone_Multiplier":":.3f"
        },
        scope="usa",
        labels={"Predicted_Price": "Price ($)"}
    )
    if isinstance(state_code, str) and len(state_code)==2 and state_code in df["Destination_State"].values:
        sel = df.loc[df["Destination_State"]==state_code].iloc[0]
        fig.add_trace(go.Choropleth(
            locations=[state_code],
            locationmode="USA-states",
            z=[sel["Predicted_Price"]],
            colorscale=[[0,"rgba(0,0,0,0)"], [1,"rgba(0,0,0,0)"]],
            showscale=False,
            marker_line_width=3,
            marker_line_color="black",
            hoverinfo="skip"
        ))
        st.metric("Predicted Price", f"${sel['Predicted_Price']:,.2f}",
                  help=f"Zone: {sel['Zone']} • Ratio: {sel['Zone_Expected_Cost_Ratio']:.3f} • Mult: {sel['Zone_Multiplier']:.3f}")
    st.plotly_chart(fig, use_container_width=True)

def metrics_cards(metrics: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Margin", f"{metrics['mean_margin']*100:.2f}%")
    c2.metric("% Inside 10–40%", f"{metrics['pct_inside']*100:.1f}%")
    c3.metric("% Below 10%", f"{metrics['pct_below']*100:.1f}%")
    c4.metric("% Above 40%", f"{metrics['pct_above']*100:.1f}%")
