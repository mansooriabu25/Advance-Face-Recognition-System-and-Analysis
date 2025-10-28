import plotly.express as px
import streamlit as st
import time, datetime
import pandas as pd
import os

VISUALS_LOG_PATH = 'visuals_log.csv' 

def BarChart():
    # --- Hourly Gender Distribution Chart ---
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
            key="barchart_filter"  # ✅ Unique key added here
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
                if "timestamp" in df.columns and "gender" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                    df = df[df["timestamp"].notna()]

                    # Create hour bins (hourly grouping)
                    df["hour_bin"] = df["timestamp"].dt.floor("h")
                    df["hour_label"] = df["hour_bin"].dt.strftime("%H:%M")

                    # Count visitors per gender per hour
                    hourly_gender_counts = (
                        df.groupby(["hour_label", "gender"])
                        .size()
                        .reset_index(name="Count")
                    )

                    # Prepare hover info — exact times of appearances
                    hover_data = (
                        df.groupby(["hour_label", "gender"])["timestamp"]
                        .apply(lambda x: ", ".join(x.dt.strftime("%H:%M:%S").tolist()))
                        .reset_index()
                        .rename(columns={"timestamp": "Exact_Times"})
                    )

                    # Merge hover info with counts
                    hourly_gender_counts = pd.merge(
                        hourly_gender_counts, hover_data, on=["hour_label", "gender"], how="left"
                    )

                    # Create grouped bar chart
                    fig_hourly_gender = px.bar(
                        hourly_gender_counts,
                        x="hour_label",
                        y="Count",
                        color="gender",
                        barmode="group", 
                        title="🕒 Hourly Gender Distribution Bar Chart",
                        hover_data=["Exact_Times"],
                        color_discrete_map={
                            "Male": "#71B4F8",
                            "Female": "#F897C8",
                            "Other": "#AAAAAA"
                        }
                    )

                    # Style chart
                    fig_hourly_gender.update_layout(
                        xaxis_title="Hour of the Day",
                        yaxis_title="Number of Visitors",
                        plot_bgcolor="black",
                        hoverlabel=dict(bgcolor="white", font_color="black"),
                        margin=dict(l=60, r=40, t=60, b=50),
                    )

                    st.plotly_chart(fig_hourly_gender, use_container_width=True)
                else:
                    st.info("Timestamp or gender column not found to plot hourly gender distribution.")
