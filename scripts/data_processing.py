import pandas as pd
import numpy as np
import io
from lichess_api import fetch_lichess_games

def fetch_and_preprocess(username, since, until, variant, analyzed=None, rated=None):
    """
    Fetch games from Lichess API and preprocess the data for visualizations.

    :param username: Lichess username
    :param since: Start date in milliseconds
    :param until: End date in milliseconds
    :param variant: Game variant (e.g., blitz, bullet)
    :param analyzed: Filter for analyzed games (True, False, or None)
    :param rated: Filter for rated games (True, False, or None)
    :return: Dictionary of temporary CSVs for each visualization
    """
    # Fetch games
    games_data = fetch_lichess_games(username, since=since, until=until, perf_type=variant, analyzed=analyzed, rated=rated)
    

    games_df = pd.DataFrame(games_data)

    if games_df.empty:
        return None  # No data found

    # Preprocess data
    csv_dict = {}

    # Elo progression
    if "WhiteElo" in games_df.columns and "BlackElo" in games_df.columns:
        games_df["UserElo"] = games_df.apply(
            lambda row: row["WhiteElo"] if row["UserColor"] == "white" else row["BlackElo"], axis=1
        )
        games_df = games_df.sort_values(by="Date")
        elo_progression = games_df[["Date", "UserElo"]]
        elo_csv = io.StringIO()
        elo_progression.to_csv(elo_csv, index=False)
        elo_csv.seek(0)
        csv_dict["elo_progression"] = elo_csv
        print(f"Elo progression dataset created with {len(elo_progression)} rows.")  # Debug: Elo progression
    else:
        print("Elo data is not available. Skipping Elo progression calculation.")  # Debug: Missing Elo data

    # Win rates and additional statistics
    valid_results = ["white", "black", "draw"]
    games_df = games_df[games_df['Result'].isin(valid_results)]

    white_games = games_df[games_df['UserColor'] == 'white']
    black_games = games_df[games_df['UserColor'] == 'black']

    # Calculate win, draw, and loss percentages for white
    white_win_rate = (white_games[white_games['Result'] == 'white'].shape[0] / white_games.shape[0]) * 100 if white_games.shape[0] > 0 else 0
    white_draw_rate = (white_games[white_games['Result'] == 'draw'].shape[0] / white_games.shape[0]) * 100 if white_games.shape[0] > 0 else 0
    white_loss_rate = (white_games[white_games['Result'] == 'black'].shape[0] / white_games.shape[0]) * 100 if white_games.shape[0] > 0 else 0

    # Calculate win, draw, and loss percentages for black
    black_win_rate = (black_games[black_games['Result'] == 'black'].shape[0] / black_games.shape[0]) * 100 if black_games.shape[0] > 0 else 0
    black_draw_rate = (black_games[black_games['Result'] == 'draw'].shape[0] / black_games.shape[0]) * 100 if black_games.shape[0] > 0 else 0
    black_loss_rate = (black_games[black_games['Result'] == 'white'].shape[0] / black_games.shape[0]) * 100 if black_games.shape[0] > 0 else 0

    # Create a DataFrame with all the rates
    win_rates = pd.DataFrame({
        "Color": ["White", "Black"],
        "Win Rate (%)": [white_win_rate, black_win_rate],
        "Draw Rate (%)": [white_draw_rate, black_draw_rate],
        "Loss Rate (%)": [white_loss_rate, black_loss_rate]
    })

    # Save to a temporary CSV
    win_rate_csv = io.StringIO()
    win_rates.to_csv(win_rate_csv, index=False)
    win_rate_csv.seek(0)
    csv_dict["win_rates"] = win_rate_csv
    print(f"Win rates dataset created with {len(win_rates)} rows.")  # Debug: Win rates

    # Games by result
    results = games_df['Result'].value_counts().reset_index()
    results.columns = ["Result", "Count"]
    result_csv = io.StringIO()
    results.to_csv(result_csv, index=False)
    result_csv.seek(0)
    csv_dict["game_results"] = result_csv

    return csv_dict