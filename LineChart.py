import plotly.express as px
import streamlit as st
import time, datetime
import pandas as pd
import os

VISUALS_LOG_PATH = 'visuals_log.csv'

def LineChart():
    visuals_path = VISUALS_LOG_PATH
    if os.path.exists(visuals_path):
        try:
            df = pd.read_csv(visuals_path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            else:
                df["timestamp"] = pd.NaT
        except Exception:
            df = pd.DataFrame(columns=["identifier", "timestamp", "datetime"])

        # --- Filter options ---
        filter_option = st.selectbox(
            "Select time range:",
            ["All time", "Today", "Last 30 minutes", "Last 1 hour", "Last 3 hours"],
            key="linechart_filter"  # ✅ Unique key added
        )

        now = datetime.datetime.now()
        if not df.empty and df["timestamp"].notna().any():
            if filter_option == "Today":
                df = df[df["timestamp"].dt.date == now.date()]
            elif filter_option == "Last 30 minutes":
                df = df[df["timestamp"] >= now - pd.Timedelta(minutes=30)]
            elif filter_option == "Last 1 hour":
                df = df[df["timestamp"] >= now - pd.Timedelta(hours=1)]
            elif filter_option == "Last 3 hours":
                df = df[df["timestamp"] >= now - pd.Timedelta(hours=3)]

        if not df.empty and df["timestamp"].notna().any():
            # --- Filter between 9 AM and 11 PM ---
            df = df[(df["timestamp"].dt.hour >= 9) & (df["timestamp"].dt.hour <= 23)]

            if not df.empty:
                # --- Create hourly bins ---
                df["hour_bin"] = df["timestamp"].dt.floor("h")

                hour_range = pd.date_range(
                    df["hour_bin"].min().replace(hour=9, minute=0, second=0),
                    df["hour_bin"].max().replace(hour=23, minute=0, second=0),
                    freq="h",
                )

                grouped = (
                    df.groupby("hour_bin")
                    .agg({
                        "identifier": lambda x: ", ".join(map(str, x)),
                        "timestamp": lambda x: ", ".join(x.dt.strftime("%H:%M:%S").tolist()),
                        "datetime": "count",
                    })
                    .reset_index()
                    .rename(columns={"datetime": "Count"})
                )

                grouped = pd.merge(
                    pd.DataFrame({"hour_bin": hour_range}),
                    grouped,
                    on="hour_bin",
                    how="left"
                ).fillna({
                    "Count": 0,
                    "identifier": "",
                    "timestamp": ""
                })

                grouped["time_range"] = grouped["hour_bin"].dt.strftime("%H:%M")

                # --- Create interactive line chart (Plotly Express) ---
                fig = px.line(
                    grouped,
                    x="time_range",
                    y="Count",
                    markers=True,
                    title="🕒 Visitors per Hour (9 AM - 11 PM) Line Chart",
                    hover_data={
                        "identifier": True,
                        "timestamp": True,
                        "Count": True,
                        "time_range": False
                    },
                )
                fig.update_traces(line=dict(width=3, color="#FC3407"), marker=dict(size=8, color="white"))

                # --- Chart Styling ---
                fig.update_layout(
                    xaxis_title="Time Range (Hour)",
                    yaxis_title="Number of Visitors",
                    plot_bgcolor="black",
                    hoverlabel=dict(bgcolor="white", font_color="black"),
                    xaxis=dict(showgrid=True, gridcolor="gray"),
                    yaxis=dict(showgrid=True, gridcolor="gray", dtick=1),
                    margin=dict(l=70, r=40, t=80, b=50),
                )

                st.plotly_chart(fig, use_container_width=True)
