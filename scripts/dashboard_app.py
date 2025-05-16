import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from data_processing import fetch_and_preprocess


def main():
    st.set_page_config(page_title="Lichess Dashboard", layout="wide")
    st.title("Lichess Dashboard")
    st.markdown(
        """
        Welcome to the **Lichess Dashboard**! This app allows you to analyze your chess games from Lichess.
        Use the sidebar to input your parameters and explore visualizations of your performance.
        """
    )

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
        "Rated Games",
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
                win_rates = pd.read_csv(csv_dict["win_rates"])
                game_results = pd.read_csv(csv_dict["game_results"])
                elo_progression = pd.read_csv(csv_dict["elo_progression"])
                opening_performance = pd.read_csv(csv_dict["opening_performance"])

                # Tabs for better organization
                tab1, tab2, tab3, tab4 = st.tabs(["Elo Progression", "Win Rates", "Games by Opening", "Opening Performance"])

                # Tab 1: Elo Progression
                with tab1:
                    st.subheader("Elo Progression with Polynomial Approximation")
                    elo_progression["Date"] = pd.to_datetime(elo_progression["Date"])
                    elo_progression = elo_progression.sort_values(by="Date")

                    # Polynomial approximation
                    x = np.arange(len(elo_progression))
                    y = elo_progression["UserElo"]
                    z = np.polyfit(x, y, 5)  # 5th-degree polynomial
                    p = np.poly1d(z)
                    elo_progression["PolynomialElo"] = p(x)

                    # Plot the graph
                    fig_elo = px.scatter(
                        elo_progression,
                        x="Date",
                        y="UserElo",
                        labels={"UserElo": "Elo", "Date": "Date"},
                        title="Elo Progression Over Time"
                    )

                    # Add the polynomial approximation as a line
                    fig_elo.add_scatter(
                        x=elo_progression["Date"],
                        y=elo_progression["PolynomialElo"],
                        mode="lines",
                        name="Polynomial Approximation"
                    )

                    # Display the chart
                    st.plotly_chart(fig_elo, use_container_width=True)

                # Tab 2: Win Rates
                with tab2:
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

                # Tab 3: Games by Opening
                with tab3:
                    st.subheader("Number of Games Played per Opening")

                    # Filter for white openings
                    white_openings = opening_performance[opening_performance["UserColor"] == "white"].nlargest(10, "TotalGames")
                    fig_white_openings_count = px.bar(
                        white_openings,
                        x="OpeningName",
                        y="TotalGames",
                        title="Top 10 Openings by Games Played (White)",
                        labels={"TotalGames": "Number of Games", "OpeningName": "Opening"},
                        text="TotalGames"
                    )
                    st.plotly_chart(fig_white_openings_count, use_container_width=True)

                    # Filter for black openings
                    black_openings = opening_performance[opening_performance["UserColor"] == "black"].nlargest(10, "TotalGames")
                    fig_black_openings_count = px.bar(
                        black_openings,
                        x="OpeningName",
                        y="TotalGames",
                        title="Top 10 Openings by Games Played (Black)",
                        labels={"TotalGames": "Number of Games", "OpeningName": "Opening"},
                        text="TotalGames"
                    )
                    st.plotly_chart(fig_black_openings_count, use_container_width=True)

                # Tab 4: Opening Performance
                with tab4:
                    st.subheader("Top 5 Openings Performance")

                    # Filter for white openings
                    white_openings = opening_performance[opening_performance["UserColor"] == "white"].nlargest(5, "TotalGames")
                    fig_white_openings = px.bar(
                        white_openings,
                        x="OpeningName",
                        y=["Win", "Draw", "Loss"],
                        title="Top 5 Openings as White",
                        labels={"value": "Percentage (%)", "OpeningName": "Opening"},
                        barmode="stack",
                        text_auto=True,
                    )
                    st.plotly_chart(fig_white_openings, use_container_width=True)

                    # Filter for black openings
                    black_openings = opening_performance[opening_performance["UserColor"] == "black"].nlargest(5, "TotalGames")
                    fig_black_openings = px.bar(
                        black_openings,
                        x="OpeningName",
                        y=["Win", "Draw", "Loss"],
                        title="Top 5 Openings as Black",
                        labels={"value": "Percentage (%)", "OpeningName": "Opening"},
                        barmode="stack",
                        text_auto=True,
                    )
                    st.plotly_chart(fig_black_openings, use_container_width=True)

    # Footer
    st.markdown(
        """
        ---
        **Lichess Dashboard** | Built with 🔥 using [Streamlit](https://streamlit.io/) and [Lichess API](https://lichess.org/api).
        """
    )


if __name__ == "__main__":
    main()