import requests
import json
import pandas as pd
from pgn_parser import parse_pgn  # Import the PGN parser function
from lichess_api import fetch_lichess_games  # Import the fetch_lichess_games function
from data_processing import fetch_and_preprocess  # Import the fetch_and_preprocess function

blitz_games = fetch_and_preprocess("mpaulinv", rated=None, since= 1514764800000, until= 1692057600000,variant="blitz", analyzed=None)
#blitz_games_df = pd.DataFrame(blitz_games)
#blitz_games_df.to_csv("mpaulinv_blitz_games.csv", index=False)



# Check and print the content of each dataset in the dictionary
if blitz_games is not None:
    for dataset_name, dataset in blitz_games.items():
        # Read the CSV-like object into a DataFrame
        dataset.seek(0)  # Reset the StringIO object to the beginning
        df = pd.read_csv(dataset)
        print(f"Dataset '{dataset_name}':")
        print(df)  # Print the DataFrame content
        print("\n" + "-" * 50 + "\n")  # Separator for readability
else:
    print("No datasets were returned.")