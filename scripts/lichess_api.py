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
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List, Dict, Optional, Any
import logging
from pgn_parser import parse_pgn  # Import the PGN parser function

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for parsed PGN data to avoid re-parsing
_pgn_cache = {}

def parse_pgn_cached(pgn_string: str) -> Dict[str, Any]:
    """
    Cached version of PGN parsing to avoid re-parsing identical PGNs
    """
    if pgn_string in _pgn_cache:
        return _pgn_cache[pgn_string]
    
    result = parse_pgn(pgn_string)
    _pgn_cache[pgn_string] = result
    return result

def process_game_chunk(games_chunk: List[Dict], username: str) -> List[Dict]:
    """
    Process a chunk of games in parallel for better performance
    """
    processed_games = []
    username_lower = username.lower()  # Cache the lowercase username
    
    for game in games_chunk:
        try:
            # Skip invalid games early
            if not isinstance(game, dict) or "pgn" not in game:
                continue
            
            # Extract basic fields with default values
            game_id = game.get("id", "Unknown")
            created_at = game.get("createdAt")
            
            # Fast player data extraction
            players = game.get("players", {})
            white_player = players.get("white", {})
            black_player = players.get("black", {})
            
            white_elo = white_player.get("rating", "Unknown")
            black_elo = black_player.get("rating", "Unknown")
            white_username = white_player.get("user", {}).get("name", "Anonymous")
            black_username = black_player.get("user", {}).get("name", "Anonymous")
            
            # Determine user color (optimized)
            user_color = "white" if white_username.lower() == username_lower else "black"
            
            # Fast move counting
            moves = game.get("moves", "")
            num_moves = len(moves.split()) if moves else 0
            
            # Extract accuracy data
            white_analysis = white_player.get("analysis", {})
            black_analysis = black_player.get("analysis", {})
            accuracy_white = white_analysis.get("accuracy") if white_analysis else None
            accuracy_black = black_analysis.get("accuracy") if black_analysis else None
            
            # Parse PGN with caching
            pgn_info = parse_pgn_cached(game["pgn"])
            
            # Build game data efficiently
            game_data = {
                "GameID": game_id,
                "Date": pd.to_datetime(created_at, unit="ms") if created_at else None,
                "Rated": game.get("rated", False),
                "Variant": game.get("variant", "Unknown"),
                "Speed": game.get("speed", "Unknown"),
                "GameType": game.get("perf", "Unknown"),
                "WhiteElo": white_elo,
                "BlackElo": black_elo,
                "WhiteUsername": white_username,
                "BlackUsername": black_username,
                "NumMoves": num_moves,
                "Result": game.get("winner", "draw"),
                "Status": game.get("status", "Unknown"),
                "AccuracyWhite": accuracy_white,
                "AccuracyBlack": accuracy_black,
                "UserColor": user_color,
                "OpeningName": pgn_info.get("OpeningName", "Unknown"),
                "OpeningECO": pgn_info.get("OpeningECO", "Unknown"),
            }
            
            processed_games.append(game_data)
            
        except Exception as e:
            logger.warning(f"Error processing game {game.get('id', 'Unknown')}: {e}")
            continue
    
    return processed_games

def fetch_lichess_games(username: str, since: Optional[int] = None, until: Optional[int] = None, 
                       perf_type: str = "blitz", max_games: Optional[int] = None, 
                       analyzed: Optional[bool] = None, rated: Optional[bool] = None) -> List[Dict]:
    """
    Fetch games for a given user from the Lichess API, including PGN data.
    Optimized for maximum performance with parallel processing and caching.

    :param username: Lichess username
    :param since: Start date (in milliseconds since epoch) or None for all games
    :param until: End date (in milliseconds since epoch) or None for all games
    :param perf_type: Game type (e.g., blitz, bullet, rapid)
    :param max_games: Maximum number of games to retrieve (optional)
    :param analyzed: If True, fetch only analyzed games; if False, fetch only non-analyzed games; if None, fetch all games
    :param rated: If True, fetch only rated games; if None, fetch all games
    :return: List of dictionaries containing game data
    """
    start_time = time.time()
    
    # Validate username
    if not username or not username.strip():
        raise ValueError("Username cannot be empty")
    
    username = username.strip()
    
    # Lichess API endpoint for games
    url = f"https://lichess.org/api/games/user/{username}"

    # Build parameters efficiently
    params = {
        "perfType": perf_type,
        "pgnInJson": True,
        "accuracy": True,
        "opening": True
    }
    
    # Add optional parameters only if they're not None
    if since is not None:
        params["since"] = since
    if until is not None:
        params["until"] = until
    if max_games is not None:
        params["max"] = max_games
    if analyzed is not None:
        params["analysed"] = analyzed
    if rated is not None:
        params["rated"] = rated

    # Optimized headers
    headers = {
        "Accept": "application/x-ndjson",
        "User-Agent": "LichessChessDashboard/1.0"
    }

    # Log request details
    logger.info(f"Fetching games for user: {username}")
    logger.info(f"URL: {url}")
    logger.info(f"Parameters: {params}")
    
    try:
        # Make request with timeout and streaming
        response = requests.get(
            url, 
            headers=headers, 
            params=params, 
            stream=True, 
            timeout=30  # 30 second timeout
        )
        
        # Enhanced error handling with specific status codes
        if response.status_code == 404:
            raise ValueError(f"User '{username}' not found on Lichess. Please check the username.")
        elif response.status_code == 429:
            raise Exception("Rate limit exceeded. Please wait a moment and try again.")
        elif response.status_code == 400:
            raise ValueError(f"Invalid request parameters: {params}")
        elif response.status_code != 200:
            error_message = f"Lichess API error: {response.status_code}"
            try:
                error_text = response.text[:200]  # First 200 chars of error
                error_message += f" - {error_text}"
            except:
                pass
            raise Exception(error_message)

        # Process response efficiently
        raw_games = []
        
        logger.info("Processing API response...")
        
        # Read and parse JSON lines efficiently
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line.strip():  # Skip empty lines
                    try:
                        game = json.loads(line)
                        raw_games.append(game)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON line: {e}")
                        continue  # Skip malformed JSON lines
        except Exception as e:
            logger.error(f"Error reading response: {e}")
            raise Exception(f"Error processing API response: {e}")
        
        logger.info(f"Fetched {len(raw_games)} raw games")
        
        if not raw_games:
            logger.warning("No games found for the specified criteria")
            return []
        
        # Process games in parallel for maximum performance
        games_data = []
        
        # Determine optimal chunk size
        num_games = len(raw_games)
        if num_games < 50:
            # For small datasets, process sequentially
            games_data = process_game_chunk(raw_games, username)
        else:
            # For larger datasets, use parallel processing
            chunk_size = max(50, num_games // 4)  # Divide into 4 chunks or minimum 50 games per chunk
            game_chunks = [raw_games[i:i + chunk_size] for i in range(0, num_games, chunk_size)]
            
            logger.info(f"Processing {len(game_chunks)} chunks in parallel...")
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=min(4, len(game_chunks))) as executor:
                # Submit all chunks for processing
                future_to_chunk = {
                    executor.submit(process_game_chunk, chunk, username): chunk 
                    for chunk in game_chunks
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    try:
                        chunk_result = future.result()
                        games_data.extend(chunk_result)
                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}")
                        continue
        
        # Log performance metrics
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Successfully processed {len(games_data)} games in {processing_time:.2f} seconds")
        if processing_time > 0:
            logger.info(f"Average processing rate: {len(games_data)/processing_time:.1f} games/second")
        
        return games_data
        
    except requests.exceptions.Timeout:
        logger.error("Request timed out after 30 seconds")
        raise Exception("Request timed out. Please try again or reduce the date range.")
    except requests.exceptions.ConnectionError:
        logger.error("Connection error - check your internet connection")
        raise Exception("Connection error. Please check your internet connection.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching games: {e}")
        raise Exception(f"Network error: {e}")
    except ValueError as e:
        # Re-raise ValueError as is (user input errors)
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise Exception(f"Unexpected error: {e}")

def clear_pgn_cache():
    """Clear the PGN parsing cache to free memory"""
    global _pgn_cache
    _pgn_cache.clear()
    logger.info("PGN cache cleared")

def get_cache_stats():
    """Get statistics about the PGN cache"""
    return {
        "cache_size": len(_pgn_cache),
        "memory_estimate_kb": len(str(_pgn_cache)) / 1024
    }

def test_api_connection(username: str) -> Dict[str, Any]:
    """
    Test if the Lichess API is accessible and the username exists
    Returns detailed status information
    """
    try:
        if not username or not username.strip():
            return {"success": False, "error": "Username cannot be empty"}
            
        url = f"https://lichess.org/api/games/user/{username.strip()}"
        response = requests.get(url, params={"max": 1}, timeout=10)
        
        if response.status_code == 200:
            return {"success": True, "message": f"User '{username}' found on Lichess"}
        elif response.status_code == 404:
            return {"success": False, "error": f"User '{username}' not found on Lichess"}
        elif response.status_code == 429:
            return {"success": False, "error": "Rate limit exceeded"}
        else:
            return {"success": False, "error": f"API returned status {response.status_code}"}
            
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Connection timed out"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Connection error - check internet"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}

# Debugging function
def debug_api_call(username: str, **kwargs):
    """
    Debug function to test API calls with detailed logging
    """
    logger.info("=== DEBUG API CALL ===")
    logger.info(f"Username: {username}")
    logger.info(f"Parameters: {kwargs}")
    
    # Test connection first
    connection_test = test_api_connection(username)
    logger.info(f"Connection test: {connection_test}")
    
    if not connection_test["success"]:
        return connection_test
    
    try:
        # Try to fetch just 1 game for testing
        result = fetch_lichess_games(username, max_games=1, **kwargs)
        logger.info(f"Debug call successful - found {len(result)} games")
        return {"success": True, "games_found": len(result)}
    except Exception as e:
        logger.error(f"Debug call failed: {e}")
        return {"success": False, "error": str(e)}