import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from data_processing import fetch_and_preprocess


def main():
    st.set_page_config(page_title="Lichess Dashboard", layout="wide")
    st.title("Lichess Dashboard")

    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    username = st.sidebar.text_input("Enter your Lichess username", value="")
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    variant = st.sidebar.selectbox("Select Game Variant", ["blitz", "bullet", "rapid", "classical"])
    analyzed_filter = st.sidebar.selectbox(
        "Filter by Analysis",
        ["All Games", "Analyzed Games Only", "Non-Analyzed Games"]
    )
    rated_filter = st.sidebar.selectbox(
        "Include Games",
        ["All Games", "Rated Games Only", "Non-Rated Games Only"]  # Dropdown for rated filter
    )

    # Fetch data button
    if st.sidebar.button("Fetch Data"):
        # Convert dates to milliseconds
        since = int(pd.Timestamp(start_date).timestamp() * 1000)
        until = int(pd.Timestamp(end_date).timestamp() * 1000)

        # Determine the analyzed parameter
        analyzed = None
        if analyzed_filter == "Analyzed Games Only":
            analyzed = True
        elif analyzed_filter == "Non-Analyzed Games":
            analyzed = False

        # Determine the rated parameter
        rated = None
        if rated_filter == "Rated Games Only":
            rated = True
        elif rated_filter == "Non-Rated Games Only":
            rated = False

        # Fetch and preprocess data
        with st.spinner("Fetching and preprocessing data..."):
            csv_dict = fetch_and_preprocess(username, since, until, variant, analyzed, rated)

            if csv_dict is None:
                st.warning("No games found for the given parameters.")
            else:
                # Load preprocessed data
                elo_progression = pd.read_csv(csv_dict["elo_progression"])
                win_rates = pd.read_csv(csv_dict["win_rates"])
                game_results = pd.read_csv(csv_dict["game_results"])

                # Layout: Columns for better organization
                col1, col2 = st.columns(2)

                # Data preview
                with col1:
                    st.subheader("Elo Progression Data Preview")
                    st.dataframe(elo_progression.head())

                # Elo progression
                with col2:
                    st.subheader("Elo Progression Over Time")
                    elo_progression["DateNumeric"] = pd.to_datetime(elo_progression["Date"]).map(pd.Timestamp.toordinal)
                    x = elo_progression["DateNumeric"]
                    y = elo_progression["UserElo"]
                    coeffs = np.polyfit(x, y, deg=3)  # Polynomial regression (degree 3)
                    poly_fit = np.poly1d(coeffs)
                    elo_progression["FittedElo"] = poly_fit(x)

                    fig_elo = px.scatter(
                        elo_progression,
                        x="Date",
                        y="UserElo",
                        title="Elo Progression Over Time (with Polynomial Fit)",
                        labels={"Date": "Date", "UserElo": "Elo"},
                    )
                    fig_elo.add_scatter(
                        x=elo_progression["Date"],
                        y=elo_progression["FittedElo"],
                        mode="lines",
                        name="Polynomial Fit",
                    )
                    st.plotly_chart(fig_elo, use_container_width=True)

                # Win, draw, and loss rates
                st.subheader("Win, Draw, and Loss Rates")

                # Prepare data for the stacked bar chart
                win_rate_stacked = win_rates.melt(
                    id_vars=["Color"],
                    value_vars=["Win Rate (%)", "Draw Rate (%)", "Loss Rate (%)"],
                    var_name="Result Type",
                    value_name="Percentage"
                )

                # Create the stacked bar chart
                fig_win_rate_stacked = px.bar(
                    win_rate_stacked,
                    x="Color",
                    y="Percentage",
                    color="Result Type",
                    title="Win, Draw, and Loss Rates by Color",
                    text="Percentage",
                    labels={"Percentage": "Percentage (%)", "Color": "Player Color"}
                )

                # Display the chart
                st.plotly_chart(fig_win_rate_stacked, use_container_width=True)

                # Games by result
                st.subheader("Games by Result")
                fig_result = px.bar(
                    game_results,
                    x="Result",
                    y="Count",
                    title="Games by Result",
                    color="Result",
                    text="Count"
                )
                st.plotly_chart(fig_result, use_container_width=True)


if __name__ == "__main__":
    main()