from lichess_api import fetch_lichess_games
from pgn_parser import parse_pgn
import pandas as pd
from datetime import datetime

def date_to_milliseconds(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)

def get_user_input():
    """Prompt the user for input dates, username, and game type, and return them."""
    username = input("Enter your Lichess username: ").strip()
    perf_type = input("Enter game type (e.g., blitz, bullet, rapid) or leave blank for 'blitz': ").strip() or "blitz"
    since_input = input("Enter start date (YYYY-MM-DD) or leave blank for all games: ").strip()
    until_input = input("Enter end date (YYYY-MM-DD) or leave blank for all games: ").strip()
    since = date_to_milliseconds(since_input) if since_input else None
    until = date_to_milliseconds(until_input) if until_input else None
    return username, perf_type, since, until

def fetch_and_process_games(username, since, until, perf_type="blitz"):
    """Fetch games from the API and process them into a DataFrame."""
    try:
        raw_games = fetch_lichess_games(username, since, until, perf_type)
        print(f"Number of games fetched: {len(raw_games)}")

        processed_games = []
        for game in raw_games:
            processed_games.append({
                "GameID": game.get("GameID", "Unknown"),
                "Date": game.get("Date", None),
                "Rated": game.get("Rated", False),
                "Variant": game.get("Variant", "Unknown"),
                "Speed": game.get("Speed", "Unknown"),
                "GameType": game.get("GameType", "Unknown"),
                "WhiteElo": game.get("WhiteElo", "Unknown"),
                "BlackElo": game.get("BlackElo", "Unknown"),
                "WhiteUsername": game.get("WhiteUsername", "Unknown"),
                "BlackUsername": game.get("BlackUsername", "Unknown"),
                "Result": game.get("Result", "Unknown"),
                "NumMoves": game.get("NumMoves", 0),
                "Status": game.get("Status", "Unknown"),
                "AccuracyWhite": game.get("AccuracyWhite", None),
                "AccuracyBlack": game.get("AccuracyBlack", None),
                "UserColor": game.get("UserColor", "Unknown"),
                "OpeningName": game.get("OpeningName", "Unknown"),
                "OpeningECO": game.get("OpeningECO", "Unknown"),
            })

        games_df = pd.DataFrame(processed_games)
        return games_df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    """Main function to orchestrate the workflow."""
    username, perf_type, since, until = get_user_input()

    games_df = fetch_and_process_games(username, since, until, perf_type)
    if games_df is not None:
        return games_df  # Return the DataFrame instead of saving it
    else:
        return None

if __name__ == "__main__":
    main()