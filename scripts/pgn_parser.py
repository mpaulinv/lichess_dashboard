import chess.pgn
from io import StringIO

def parse_pgn(pgn_data):
    """
    Parse the PGN data of a game to extract additional information.

    :param pgn_data: Raw PGN string of the game
    :return: Dictionary with extracted PGN information 
    """
    if not pgn_data:
        return {"OpeningName": "Unknown", "OpeningECO": "Unknown"}

    try:
        pgn_io = StringIO(pgn_data)
        game_pgn = chess.pgn.read_game(pgn_io)
        

        if not game_pgn:
            print("Failed to parse PGN.")
            return {"OpeningECO": "Unknown"}

        # Extract ECO code from PGN headers
        opening_eco = game_pgn.headers.get("ECO", "Unknown")
        opening_name = game_pgn.headers.get("Opening", "Unknown")
        return {"OpeningName": opening_name, "OpeningECO": opening_eco}
    except Exception as e:
        print(f"Error parsing PGN: {e}")
        return {"OpeningName": "Unknown", "OpeningECO": "Unknown"}
