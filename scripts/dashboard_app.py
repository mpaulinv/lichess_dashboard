import pandas as pd
import streamlit as st
import plotly.express as px
from lichess_api import fetch_lichess_games

# Calculate win rates
def calculate_win_rates(data):
    white_games = data[data['UserColor'] == 'white']
    black_games = data[data['UserColor'] == 'black']

    white_wins = white_games[white_games['Result'] == 'white'].shape[0]
    black_wins = black_games[black_games['Result'] == 'black'].shape[0]

    total_white_games = white_games.shape[0]
    total_black_games = black_games.shape[0]

    white_win_rate = (white_wins / total_white_games) * 100 if total_white_games > 0 else 0
    black_win_rate = (black_wins / total_black_games) * 100 if total_black_games > 0 else 0

    return white_win_rate, black_win_rate

# Calculate games by result
def calculate_games_by_result(data):
    results = data['Result'].value_counts()
    return results

# Streamlit app
def main():
    st.title("Lichess Dashboard")

    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    username = st.sidebar.text_input("Enter your Lichess username", value="mpaulinv")
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    variant = st.sidebar.selectbox("Select Game Variant", ["blitz", "bullet", "rapid", "classical"])

    # Fetch data button
    if st.sidebar.button("Fetch Data"):
        # Convert dates to milliseconds
        since = int(pd.Timestamp(start_date).timestamp() * 1000)
        until = int(pd.Timestamp(end_date).timestamp() * 1000)

        # Fetch games
        with st.spinner("Fetching games..."):
            try:
                games_data = fetch_lichess_games(username, since, until, variant)
                games_df = pd.DataFrame(games_data)

                if games_df.empty:
                    st.warning("No games found for the given parameters.")
                else:
                    # Display dashboards
                    st.write(f"### Chess Dashboard for {username}")

                    # Data preview
                    st.write("### Data Preview")
                    st.dataframe(games_df.head())

                    # Win rates
                    white_win_rate, black_win_rate = calculate_win_rates(games_df)
                    st.write(f"### Win Rates")
                    st.write(f"**White Win Rate:** {white_win_rate:.2f}%")
                    st.write(f"**Black Win Rate:** {black_win_rate:.2f}%")

                    # Win rate bar chart
                    win_rate_data = pd.DataFrame({
                        "Color": ["White", "Black"],
                        "Win Rate (%)": [white_win_rate, black_win_rate]
                    })
                    fig_win_rate = px.bar(win_rate_data, x="Color", y="Win Rate (%)", title="Win Rate by Color", color="Color")
                    st.plotly_chart(fig_win_rate)

                    # Games by result
                    results = calculate_games_by_result(games_df)
                    st.write(f"### Games by Result")
                    result_data = pd.DataFrame({
                        "Result": results.index,
                        "Count": results.values
                    })
                    fig_result = px.bar(result_data, x="Result", y="Count", title="Games by Result", color="Result")
                    st.plotly_chart(fig_result)

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()