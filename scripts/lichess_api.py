"""
lichess_api.py

This module provides functionality to interact with the Lichess API to fetch chess games for a given user.
It allows users to specify a username, timeframe, and game type to retrieve games in bulk. The fetched
games are processed and returned as a pandas DataFrame for further analysis.

Features:
- Fetch games for a specific user within a given timeframe.
- Filter games by performance type (e.g., blitz, bullet, rapid).
- Extract relevant game details such as player ratings, game type, result, and number of moves.
- Return the processed data as a pandas DataFrame for easy manipulation and analysis.

Dependencies:
- requests: For making HTTP requests to the Lichess API.
- pandas: For organizing and processing the game data.
- json: For parsing the JSON responses from the API.

Usage:
- Import the `fetch_lichess_games` function and call it with the desired parameters.
- Example:
    from lichess_api import fetch_lichess_games
    games_df = fetch_lichess_games("username", since_timestamp, until_timestamp, "blitz", 100)
    print(games_df)
"""

import requests
import json
import pandas as pd
from pgn_parser import parse_pgn  # Import the PGN parser function

def fetch_lichess_games(username, since=None, until=None, perf_type="blitz", max_games=None, analyzed=None, rated=None):
    """
    Fetch games for a given user from the Lichess API, including PGN data.

    :param username: Lichess username
    :param since: Start date (in milliseconds since epoch) or None for all games
    :param until: End date (in milliseconds since epoch) or None for all games
    :param perf_type: Game type (e.g., blitz, bullet, rapid)
    :param max_games: Maximum number of games to retrieve (optional)
    :param analyzed: If True, fetch only analyzed games; if False, fetch only non-analyzed games; if None, fetch all games
    :param rated: If True, fetch only rated games; if None, fetch all games
    :return: List of dictionaries containing game data
    """
    # Lichess API endpoint for games
    url = f"https://lichess.org/api/games/user/{username}"

    # Specify filters
    params = {
        "perfType": perf_type,
        "pgnInJson": True,  # Include PGN in the response
        "accuracy": True,   # Include accuracy data
        "opening": True     # Include opening data
    }
    if since:
        params["since"] = since
    if until:
        params["until"] = until
    if max_games:
        params["max"] = max_games
    if analyzed is not None:
        params["analysed"] = analyzed
    if rated is not None:
        params["rated"] = rated

    # Headers for the request
    headers = {"Accept": "application/x-ndjson"}

    # Fetch data
    print(f"Fetching games for user: {username}")
    print(f"API URL: {url}")
    print(f"Parameters: {params}")
    response = requests.get(url, headers=headers, params=params, stream=True)

    # Check response status
    if response.status_code != 200:
        raise Exception(f"Error fetching games: {response.status_code} - {response.text}")

    # Initialize an empty list to store game data
    games_data = []

    # Process each game in the ndjson response
    for line in response.iter_lines():
        if line:  # Ignore empty lines
            try:
                game = json.loads(line.decode("utf-8"))  # Parse JSON
            except json.JSONDecodeError as e:
                continue

            # Ensure `game` is a dictionary
            if not isinstance(game, dict):
                continue

            # Skip unfinished games
            #if game.get("status") not in ["mate", "resign", "draw", "timeout", "stalemate", "aborted"]:
            #    continue
            # Check if the "pgn" key exists
            if "pgn" not in game:
                continue

            # Extract relevant fields
            game_id = game.get("id", "Unknown")
            created_at = game.get("createdAt", None)
            rated = game.get("rated", False)
            variant = game.get("variant", "Unknown")
            speed = game.get("speed", "Unknown")
            game_type = game.get("perf", "Unknown")
            white_elo = game["players"]["white"].get("rating", "Unknown")
            black_elo = game["players"]["black"].get("rating", "Unknown")
            white_username = game["players"]["white"].get("user", {}).get("name", "Anonymous")
            black_username = game["players"]["black"].get("user", {}).get("name", "Anonymous")
            num_moves = len(game["moves"].split())
            result = game.get("winner", "draw")  # 'draw' if no winner key exists
            status = game.get("status", "Unknown")
            accuracy_white = game.get("players", {}).get("white", {}).get("analysis", {}).get("accuracy", None)
            accuracy_black = game.get("players", {}).get("black", {}).get("analysis", {}).get("accuracy", None)

            # Determine the color the user played as
            user_color = "white" if white_username.lower() == username.lower() else "black"

            # Extract PGN-specific attributes
            pgn_info = parse_pgn(game["pgn"])  # Use the PGN parser to extract attributes like OpeningName

            # Append the extracted data
            games_data.append({
                "GameID": game_id,
                "Date": pd.to_datetime(created_at, unit="ms") if created_at else None,
                "Rated": rated,
                "Variant": variant,
                "Speed": speed,
                "GameType": game_type,
                "WhiteElo": white_elo,
                "BlackElo": black_elo,
                "WhiteUsername": white_username,
                "BlackUsername": black_username,
                "NumMoves": num_moves,
                "Result": result,
                "Status": status,
                "AccuracyWhite": accuracy_white,
                "AccuracyBlack": accuracy_black,
                "UserColor": user_color,
                "OpeningName": pgn_info.get("OpeningName", "Unknown"),  # Extracted from PGN
                "OpeningECO": pgn_info.get("OpeningECO", "Unknown"),  # Extracted from PGN
            })

    return games_data