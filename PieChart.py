import plotly.express as px
import streamlit as st
import time, datetime
import pandas as pd
import os

VISUALS_LOG_PATH = 'visuals_log.csv' 

def PieChart():
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
            key="piechart_filter"  # ✅ Unique key added here
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

                if "age" in df.columns:
                    # Ensure numeric ages (in case some entries are missing or text)
                    df["age"] = pd.to_numeric(df["age"], errors="coerce")
                    df = df[df["age"].notna()]

                    # Define age bins and labels
                    bins = [10, 20, 30, 40, 50, 60, 100]
                    labels = ["10–20 y/o", "20–30 y/o", "30–40 y/o", "40–50 y/o", "50–60 y/o", "60+ y/o"]

                    # Categorize into age groups
                    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

                    # Count occurrences
                    age_group_counts = (
                        df["age_group"].value_counts()
                        .reset_index()
                        .rename(columns={"index": "Age Group", "age_group": "Count"})
                    )
                    age_group_counts.columns = ["Age Group", "Count"]

                    # Create pie chart
                    fig_age_pie = px.pie(
                        age_group_counts,
                        names="Age Group",
                        values="Count",
                        title="🎂 Age Group Distribution Pie Chart",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )

                    # Style chart
                    fig_age_pie.update_traces(
                        textinfo="percent+label",
                        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}",
                        pull=[0.03] * len(age_group_counts)
                    )
                    fig_age_pie.update_layout(
                        showlegend=True,
                        legend_title="Age Group",
                        margin=dict(l=40, r=40, t=60, b=60)
                    )

                    st.plotly_chart(fig_age_pie, use_container_width=True)
