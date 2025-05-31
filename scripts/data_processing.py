import pandas as pd
import numpy as np
import io
from typing import Dict, Optional, Any, Union
import logging
from concurrent.futures import ThreadPoolExecutor
from lichess_api import fetch_lichess_games

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_games_dataframe(games_df: pd.DataFrame, sample_size: int = 5) -> None:
    """
    Debug function to examine the structure of the games DataFrame
    """
    print("=== GAMES DATAFRAME DEBUG ===")
    print(f"Shape: {games_df.shape}")
    print(f"Columns: {list(games_df.columns)}")
    
    # Check for opening-related columns
    opening_cols = [col for col in games_df.columns if any(keyword in col.lower() for keyword in ['opening', 'eco'])]
    print(f"Opening-related columns: {opening_cols}")
    
    # Sample data
    print(f"\nSample data (first {sample_size} rows):")
    print(games_df.head(sample_size).to_string())
    
    # Check data types
    print(f"\nData types:")
    for col in games_df.columns:
        print(f"{col}: {games_df[col].dtype}")
    
    # Check for null values in key columns
    key_columns = ['UserColor', 'Result'] + opening_cols
    print(f"\nNull values in key columns:")
    for col in key_columns:
        if col in games_df.columns:
            null_count = games_df[col].isnull().sum()
            print(f"{col}: {null_count} nulls out of {len(games_df)}")
    
    # Check unique values in key columns
    print(f"\nUnique values in key columns:")
    for col in ['UserColor', 'Result']:
        if col in games_df.columns:
            unique_vals = games_df[col].unique()
            print(f"{col}: {unique_vals}")
    
    # Check opening data specifically
    for col in opening_cols:
        if col in games_df.columns:
            non_null_openings = games_df[games_df[col].notna()]
            print(f"\n{col} stats:")
            print(f"  Non-null count: {len(non_null_openings)}")
            print(f"  Unique openings: {non_null_openings[col].nunique()}")
            if len(non_null_openings) > 0:
                print(f"  Sample openings: {non_null_openings[col].unique()[:5]}")

def fetch_and_preprocess(username: str, since: int, until: int, variant: str, 
                        analyzed: Optional[bool] = None, rated: Optional[bool] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch games from Lichess API and preprocess the data for visualizations.
    Optimized for maximum performance with parallel processing and efficient data handling.

    :param username: Lichess username
    :param since: Start date in milliseconds
    :param until: End date in milliseconds
    :param variant: Game variant (e.g., blitz, bullet)
    :param analyzed: Filter for analyzed games (True, False, or None)
    :param rated: Filter for rated games (True, False, or None)
    :return: Dictionary of processed data for each visualization
    """
    try:
        logger.info(f"Starting data fetch and preprocessing for user: {username}")
        
        # Fetch games with error handling
        games_data = fetch_lichess_games(
            username, 
            since=since, 
            until=until, 
            perf_type=variant, 
            analyzed=analyzed, 
            rated=rated
        )
        
        if not games_data:
            logger.warning("No games data retrieved from API")
            return None
        
        # Convert to DataFrame with optimized dtypes
        games_df = pd.DataFrame(games_data)
        logger.info(f"Created DataFrame with {len(games_df)} games")
        
        if games_df.empty:
            logger.warning("Empty DataFrame after conversion")
            return None

        # DEBUG: Examine the raw data structure
        logger.info("=== DEBUG: Raw games data structure ===")
        debug_games_dataframe(games_df, sample_size=3)

        # Optimize DataFrame dtypes for better performance
        games_df = optimize_dataframe_types(games_df)
        
        # Calculate overall win statistics from the main DataFrame
        overall_stats = calculate_overall_win_statistics(games_df)
        
        # Process data in parallel for better performance
        csv_dict = {}
        
        # Use ThreadPoolExecutor for parallel processing of different datasets
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit tasks for parallel processing
            futures = {
                'elo_progression': executor.submit(process_elo_progression, games_df),
                'win_rates': executor.submit(process_win_rates, games_df),
                'game_results': executor.submit(process_game_results, games_df),
                'opening_performance': executor.submit(process_opening_performance, games_df)
            }
            
            # Collect results as they complete
            for key, future in futures.items():
                try:
                    result = future.result()
                    if result is not None:
                        csv_dict[key] = result
                        logger.info(f"Successfully processed {key}: {len(result) if isinstance(result, list) else 'N/A'} records")
                    else:
                        logger.warning(f"No data for {key}")
                except Exception as e:
                    logger.error(f"Error processing {key}: {e}")
                    continue
        
        # Add overall statistics for dashboard summary
        csv_dict['overall_stats'] = overall_stats
        
        # Add polynomial approximation for Elo progression if available
        if 'elo_progression' in csv_dict and csv_dict['elo_progression'] is not None:
            try:
                poly_approx = create_polynomial_approximation(csv_dict['elo_progression'])
                if poly_approx is not None:
                    csv_dict['poly_approx'] = poly_approx
                    logger.info("Added polynomial approximation")
            except Exception as e:
                logger.warning(f"Could not create polynomial approximation: {e}")
        
        logger.info(f"Data preprocessing completed. Generated {len(csv_dict)} datasets")
        
        # Final debug: show what we generated
        logger.info("=== Final datasets summary ===")
        for key, data in csv_dict.items():
            if isinstance(data, list):
                logger.info(f"{key}: {len(data)} records")
            elif isinstance(data, dict):
                logger.info(f"{key}: dictionary with keys {list(data.keys())}")
            else:
                logger.info(f"{key}: {type(data)}")
        
        return csv_dict
        
    except Exception as e:
        logger.error(f"Error in fetch_and_preprocess: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def optimize_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame data types for better memory usage and performance
    """
    try:
        # Convert date column to datetime if it's not already
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Optimize numeric columns
        numeric_columns = ['WhiteElo', 'BlackElo', 'NumMoves']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
        
        # Optimize categorical columns
        categorical_columns = ['Result', 'UserColor', 'Variant', 'Speed', 'Status', 'OpeningName', 'OpeningECO']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Optimize boolean columns
        boolean_columns = ['Rated']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype('bool')
        
        return df
    except Exception as e:
        logger.warning(f"Error optimizing DataFrame types: {e}")
        return df

def calculate_overall_win_statistics(games_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate overall win statistics from user perspective
    This is separate from opening performance to avoid breaking charts
    """
    try:
        if games_df.empty or 'Result' not in games_df.columns or 'UserColor' not in games_df.columns:
            return {"win_rate": 0.0, "draw_rate": 0.0, "loss_rate": 0.0}
        
        # Map results to user perspective
        def map_user_result(row):
            if pd.isna(row['Result']) or pd.isna(row['UserColor']):
                return None
            
            result = str(row['Result']).lower()
            user_color = str(row['UserColor']).lower()
            
            if result == 'draw':
                return 'draw'
            elif (result == 'white' and user_color == 'white') or (result == 'black' and user_color == 'black'):
                return 'win'
            elif (result == 'white' and user_color == 'black') or (result == 'black' and user_color == 'white'):
                return 'loss'
            else:
                return None
        
        # Apply the mapping
        games_copy = games_df.copy()
        games_copy['UserResult'] = games_copy.apply(map_user_result, axis=1)
        
        # Filter out invalid results
        valid_games = games_copy[games_copy['UserResult'].notna()]
        
        if valid_games.empty:
            return {"win_rate": 0.0, "draw_rate": 0.0, "loss_rate": 0.0}
        
        # Calculate overall statistics
        total_games = len(valid_games)
        wins = len(valid_games[valid_games['UserResult'] == 'win'])
        draws = len(valid_games[valid_games['UserResult'] == 'draw'])
        losses = len(valid_games[valid_games['UserResult'] == 'loss'])
        
        return {
            "win_rate": (wins / total_games) * 100,
            "draw_rate": (draws / total_games) * 100,
            "loss_rate": (losses / total_games) * 100
        }
        
    except Exception as e:
        logger.error(f"Error calculating overall win statistics: {e}")
        return {"win_rate": 0.0, "draw_rate": 0.0, "loss_rate": 0.0}

def process_elo_progression(games_df: pd.DataFrame) -> Optional[list]:
    """
    Process Elo progression data efficiently
    Returns a list of dictionaries for easy JSON serialization
    """
    try:
        if games_df.empty or "WhiteElo" not in games_df.columns or "BlackElo" not in games_df.columns:
            logger.warning("Elo data not available")
            return None
        
        # Vectorized operation for better performance
        games_df_copy = games_df.copy()
        games_df_copy["UserElo"] = np.where(
            games_df_copy["UserColor"] == "white",
            games_df_copy["WhiteElo"],
            games_df_copy["BlackElo"]
        )
        
        # Filter out invalid Elo ratings
        games_df_copy = games_df_copy[pd.to_numeric(games_df_copy["UserElo"], errors='coerce').notna()]
        games_df_copy["UserElo"] = pd.to_numeric(games_df_copy["UserElo"])
        
        # Sort by date efficiently
        games_df_copy = games_df_copy.sort_values(by="Date")
        
        # Select only required columns
        elo_progression = games_df_copy[["Date", "UserElo"]].reset_index(drop=True)
        
        if elo_progression.empty:
            logger.warning("No valid Elo data found")
            return None
        
        logger.info(f"Elo progression dataset created with {len(elo_progression)} rows")
        
        # Return as list of dictionaries for compatibility
        return elo_progression.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error processing Elo progression: {e}")
        return None

def process_win_rates(games_df: pd.DataFrame) -> Optional[list]:
    """
    Process win rates data efficiently using vectorized operations
    Returns a list of dictionaries for easy JSON serialization
    """
    try:
        if games_df.empty:
            return None
        
        # Check for required columns
        if 'Result' not in games_df.columns or 'UserColor' not in games_df.columns:
            logger.warning("Required columns (Result, UserColor) not found for win rate calculation")
            return None
        
        # Filter valid results efficiently - map results to user perspective
        filtered_df = games_df.copy()
        
        # Create user result mapping based on color and game result
        def map_user_result(row):
            if pd.isna(row['Result']) or pd.isna(row['UserColor']):
                return None
            
            result = str(row['Result']).lower()
            user_color = str(row['UserColor']).lower()
            
            if result == 'draw':
                return 'draw'
            elif (result == 'white' and user_color == 'white') or (result == 'black' and user_color == 'black'):
                return 'win'
            elif (result == 'white' and user_color == 'black') or (result == 'black' and user_color == 'white'):
                return 'loss'
            else:
                return None
        
        # Apply the mapping
        filtered_df['UserResult'] = filtered_df.apply(map_user_result, axis=1)
        
        # Filter out invalid results
        filtered_df = filtered_df[filtered_df['UserResult'].notna()]
        
        if filtered_df.empty:
            logger.warning("No valid game results found after mapping")
            return None
        
        # Calculate statistics by color
        win_rates_data = []
        
        for color in ['white', 'black']:
            color_games = filtered_df[filtered_df['UserColor'] == color]
            
            if len(color_games) > 0:
                total_games = len(color_games)
                wins = len(color_games[color_games['UserResult'] == 'win'])
                draws = len(color_games[color_games['UserResult'] == 'draw'])
                losses = len(color_games[color_games['UserResult'] == 'loss'])
                
                win_rate = (wins / total_games) * 100
                draw_rate = (draws / total_games) * 100
                loss_rate = (losses / total_games) * 100
                
                win_rates_data.append({
                    "Color": color.capitalize(),
                    "Win Rate (%)": round(float(win_rate), 2),
                    "Draw Rate (%)": round(float(draw_rate), 2),
                    "Loss Rate (%)": round(float(loss_rate), 2),
                    "Total Games": int(total_games)
                })
        
        if not win_rates_data:
            logger.warning("No win rate data could be calculated")
            return None
        
        logger.info(f"Win rates dataset created with {len(win_rates_data)} rows")
        return win_rates_data
        
    except Exception as e:
        logger.error(f"Error processing win rates: {e}")
        return None

def process_game_results(games_df: pd.DataFrame) -> Optional[list]:
    """
    Process game results data efficiently
    Returns a list of dictionaries for easy JSON serialization
    """
    try:
        if games_df.empty or 'Result' not in games_df.columns or 'UserColor' not in games_df.columns:
            return None
        
        # Map results to user perspective
        def map_user_result(row):
            if pd.isna(row['Result']) or pd.isna(row['UserColor']):
                return 'Unknown'
            
            result = str(row['Result']).lower()
            user_color = str(row['UserColor']).lower()
            
            if result == 'draw':
                return 'Draw'
            elif (result == 'white' and user_color == 'white') or (result == 'black' and user_color == 'black'):
                return 'Win'
            elif (result == 'white' and user_color == 'black') or (result == 'black' and user_color == 'white'):
                return 'Loss'
            else:
                return 'Unknown'
        
        # Apply the mapping
        games_df_copy = games_df.copy()
        games_df_copy['UserResult'] = games_df_copy.apply(map_user_result, axis=1)
        
        # Use value_counts for efficient counting
        results = games_df_copy['UserResult'].value_counts().reset_index()
        results.columns = ["Result", "Count"]
        
        if results.empty:
            return None
        
        logger.info(f"Game results dataset created with {len(results)} rows")
        return results.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error processing game results: {e}")
        return None

def process_opening_performance(games_df: pd.DataFrame) -> Optional[list]:
    """
    Process opening performance data efficiently
    Returns a list of dictionaries for easy JSON serialization
    Keep original color-based results for opening charts
    """
    try:
        logger.info(f"Processing opening performance for {len(games_df)} games")
        
        if games_df.empty:
            logger.warning("Empty games DataFrame for opening performance")
            return None
        
        # Debug: Print available columns
        logger.info(f"Available columns: {list(games_df.columns)}")
        
        # Check what opening columns we have
        opening_columns = [col for col in games_df.columns if 'opening' in col.lower() or 'eco' in col.lower()]
        logger.info(f"Opening-related columns: {opening_columns}")
        
        # Try different possible opening column names
        opening_column = None
        possible_opening_names = ['OpeningName', 'opening_name', 'Opening', 'opening', 'OpeningECO', 'opening_eco']
        
        for col_name in possible_opening_names:
            if col_name in games_df.columns:
                opening_column = col_name
                logger.info(f"Found opening column: {col_name}")
                break
        
        if opening_column is None:
            logger.warning(f"No opening column found. Available columns: {list(games_df.columns)}")
            return None
        
        # Filter out games without opening names
        filtered_df = games_df[
            games_df[opening_column].notna() & 
            (games_df[opening_column] != "Unknown") &
            (games_df[opening_column] != "") &
            (games_df[opening_column].astype(str).str.strip() != "")
        ].copy()
        
        logger.info(f"Games with opening data: {len(filtered_df)} out of {len(games_df)}")
        
        if filtered_df.empty:
            logger.warning("No games with valid opening data found")
            return None
        
        # Check for required columns
        required_columns = ['Result', 'UserColor']
        missing_columns = [col for col in required_columns if col not in filtered_df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}. Available: {list(filtered_df.columns)}")
            return None
        
        # Debug: Show sample of data
        logger.info("Sample of filtered data:")
        logger.info(filtered_df[['UserColor', opening_column, 'Result']].head().to_string())
        
        # Calculate opening performance - keep original color-based approach
        opening_performance_list = []
        
        for user_color in ['white', 'black']:
            color_games = filtered_df[filtered_df['UserColor'] == user_color]
            logger.info(f"Games as {user_color}: {len(color_games)}")
            
            if len(color_games) > 0:
                # Group by opening
                opening_groups = color_games.groupby(opening_column)
                logger.info(f"Number of unique openings as {user_color}: {len(opening_groups)}")
                
                for opening_name, group in opening_groups:
                    total_games = len(group)
                    
                    # Only include openings with at least 2 games for meaningful statistics
                    if total_games < 2:
                        continue
                    
                    # Count results based on actual game results (not user perspective)
                    white_wins = len(group[group['Result'].astype(str).str.lower() == 'white'])
                    black_wins = len(group[group['Result'].astype(str).str.lower() == 'black'])
                    draws = len(group[group['Result'].astype(str).str.lower() == 'draw'])
                    
                    # Calculate percentages
                    white_win_rate = (white_wins / total_games) * 100
                    black_win_rate = (black_wins / total_games) * 100
                    draw_rate = (draws / total_games) * 100
                    
                    # For opening charts, we want Win/Draw/Loss from the color's perspective
                    if user_color == 'white':
                        win_rate = white_win_rate
                        loss_rate = black_win_rate
                    else:  # user_color == 'black'
                        win_rate = black_win_rate
                        loss_rate = white_win_rate
                    
                    opening_performance_list.append({
                        'UserColor': user_color,
                        'OpeningName': str(opening_name),
                        'TotalGames': int(total_games),
                        'Win': round(float(win_rate), 2),
                        'Draw': round(float(draw_rate), 2),
                        'Loss': round(float(loss_rate), 2)
                    })
                    
                    logger.debug(f"Added opening: {opening_name} ({user_color}) - {total_games} games")
        
        if not opening_performance_list:
            logger.warning("No opening performance data could be calculated")
            # Debug: Let's see what we have
            logger.info("Debug info:")
            logger.info(f"Unique user colors: {filtered_df['UserColor'].unique()}")
            logger.info(f"Unique results: {filtered_df['Result'].unique()}")
            logger.info(f"Unique openings: {filtered_df[opening_column].unique()[:10]}")  # First 10
            return None
        
        logger.info(f"Opening performance dataset created with {len(opening_performance_list)} rows")
        return opening_performance_list
        
    except Exception as e:
        logger.error(f"Error processing opening performance: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def create_polynomial_approximation(elo_data: list, degree: int = 3) -> Optional[list]:
    """
    Create polynomial approximation for Elo progression
    Returns a list of dictionaries for compatibility
    """
    try:
        if not elo_data or len(elo_data) < degree + 1:
            logger.warning(f"Not enough data points for degree {degree} polynomial")
            return None
        
        # Convert to DataFrame for processing
        elo_df = pd.DataFrame(elo_data)
        
        if 'Date' not in elo_df.columns or 'UserElo' not in elo_df.columns:
            logger.warning("Required columns missing for polynomial approximation")
            return None
        
        # Convert dates to numeric values for polynomial fitting
        elo_df['Date'] = pd.to_datetime(elo_df['Date'])
        elo_df = elo_df.sort_values('Date')
        
        # Create numeric representation of dates
        date_numeric = (elo_df['Date'] - elo_df['Date'].min()).dt.days
        
        # Fit polynomial
        coefficients = np.polyfit(date_numeric, elo_df['UserElo'], degree)
        polynomial = np.poly1d(coefficients)
        
        # Generate smooth curve
        x_smooth = np.linspace(date_numeric.min(), date_numeric.max(), len(elo_df) * 2)
        y_smooth = polynomial(x_smooth)
        
        # Convert back to dates
        dates_smooth = elo_df['Date'].min() + pd.to_timedelta(x_smooth, unit='D')
        
        # Create result as list of dictionaries
        poly_data = [
            {
                'date': date.isoformat(),
                'rating': float(rating)
            }
            for date, rating in zip(dates_smooth, y_smooth)
        ]
        
        logger.info(f"Polynomial approximation created with {len(poly_data)} points")
        return poly_data
        
    except Exception as e:
        logger.error(f"Error creating polynomial approximation: {e}")
        return None

def get_processing_stats(csv_dict: Dict[str, Any]) -> Dict[str, int]:
    """
    Get statistics about processed data
    """
    stats = {}
    for key, data in csv_dict.items():
        try:
            if isinstance(data, list):
                stats[key] = len(data)
            elif isinstance(data, pd.DataFrame):
                stats[key] = len(data)
            elif isinstance(data, dict):
                # For overall_stats and similar dict entries
                stats[key] = 1
            elif isinstance(data, io.StringIO):
                data.seek(0)
                df = pd.read_csv(data)
                stats[key] = len(df)
                data.seek(0)  # Reset for future use
            else:
                stats[key] = 0
        except Exception as e:
            logger.warning(f"Error getting stats for {key}: {e}")
            stats[key] = 0
    
    return stats

# Utility function for debugging
def debug_preprocessing(username: str, **kwargs) -> Dict[str, Any]:
    """
    Debug function to test preprocessing with detailed information
    """
    try:
        result = fetch_and_preprocess(username, **kwargs)
        if result:
            stats = get_processing_stats(result)
            return {"success": True, "datasets": list(result.keys()), "stats": stats}
        else:
            return {"success": False, "error": "No data returned"}
    except Exception as e:
        logger.error(f"Debug preprocessing error: {e}")
        return {"success": False, "error": str(e)}

# Helper function to convert data for dashboard compatibility
def ensure_dataframe_compatibility(data: Any) -> pd.DataFrame:
    """
    Convert various data formats to DataFrame for dashboard compatibility
    """
    try:
        if isinstance(data, list):
            if not data:  # Empty list
                return pd.DataFrame()
            return pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, io.StringIO):
            data.seek(0)
            df = pd.read_csv(data)
            data.seek(0)  # Reset for future use
            return df
        else:
            logger.warning(f"Unknown data type for conversion: {type(data)}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error in ensure_dataframe_compatibility: {e}")
        return pd.DataFrame()

# Backward compatibility function for existing dashboard code
def get_dataframe_from_result(csv_dict: Dict[str, Any], key: str) -> Optional[pd.DataFrame]:
    """
    Get DataFrame from result dictionary with backward compatibility
    """
    if not csv_dict or key not in csv_dict or csv_dict[key] is None:
        return None
    
    try:
        df = ensure_dataframe_compatibility(csv_dict[key])
        if df.empty:
            logger.warning(f"Empty DataFrame for key: {key}")
            return None
        return df
    except Exception as e:
        logger.error(f"Error converting {key} to DataFrame: {e}")
        return None

# Additional utility functions for data validation
def validate_data_structure(csv_dict: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate the structure of processed data
    """
    validation_results = {}
    
    # Expected columns for each dataset
    expected_columns = {
        'elo_progression': ['Date', 'UserElo'],
        'win_rates': ['Color', 'Win Rate (%)', 'Draw Rate (%)', 'Loss Rate (%)'],
        'game_results': ['Result', 'Count'],
        'opening_performance': ['UserColor', 'OpeningName', 'TotalGames'],
        'poly_approx': ['date', 'rating']
    }
    
    for key, expected_cols in expected_columns.items():
        if key in csv_dict and csv_dict[key] is not None:
            df = get_dataframe_from_result(csv_dict, key)
            if df is not None:
                has_required_cols = all(col in df.columns for col in expected_cols)
                validation_results[key] = has_required_cols
                if not has_required_cols:
                    logger.warning(f"Missing required columns in {key}: {expected_cols}")
            else:
                validation_results[key] = False
        else:
            validation_results[key] = False
    
    return validation_results

def get_data_summary(csv_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a comprehensive summary of all processed data
    """
    summary = {
        'datasets': list(csv_dict.keys()) if csv_dict else [],
        'stats': get_processing_stats(csv_dict) if csv_dict else {},
        'validation': validate_data_structure(csv_dict) if csv_dict else {},
        'total_records': 0
    }
    
    if csv_dict:
        # Don't count non-list items in total records
        total_records = 0
        for key, stat in summary['stats'].items():
            if key not in ['overall_stats']:  # Skip non-list items
                total_records += stat
        summary['total_records'] = total_records
    
    return summary

def debug_data_contents(csv_dict: Dict[str, Any]) -> None:
    """
    Debug function to print the contents and structure of each dataset
    """
    print("=== DEBUG DATA CONTENTS ===")
    
    if not csv_dict:
        print("No data in csv_dict")
        return
    
    for key, data in csv_dict.items():
        print(f"\n--- {key} ---")
        print(f"Type: {type(data)}")
        
        if isinstance(data, list):
            print(f"Length: {len(data)}")
            if data:
                print(f"First item: {data[0]}")
                if isinstance(data[0], dict):
                    print(f"Keys: {list(data[0].keys())}")
        
        elif isinstance(data, pd.DataFrame):
            print(f"Shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            if not data.empty:
                print(f"First row: {data.iloc[0].to_dict()}")
        
        elif isinstance(data, dict):
            print(f"Dictionary keys: {list(data.keys())}")
            print(f"Dictionary values: {data}")
        
        else:
            print(f"Data: {str(data)[:100]}...")

def get_actual_game_count(csv_dict: Dict[str, Any]) -> int:
    """
    Get the actual number of games from the most reliable source
    """
    # Elo progression gives us the most accurate game count
    elo_data = get_dataframe_from_result(csv_dict, "elo_progression")
    if elo_data is not None and not elo_data.empty:
        return len(elo_data)
    
    # Fallback to game results
    game_results = get_dataframe_from_result(csv_dict, "game_results")
    if game_results is not None and not game_results.empty and 'Count' in game_results.columns:
        return int(game_results['Count'].sum())
    
    return 0

def get_actual_win_stats(csv_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Get actual win statistics from processed data
    """
    # First try to get from overall_stats (most accurate)
    if 'overall_stats' in csv_dict and csv_dict['overall_stats']:
        return csv_dict['overall_stats']
    
    # Fallback to win_rates data
    win_rate_data = get_dataframe_from_result(csv_dict, "win_rates")
    
    if win_rate_data is None or win_rate_data.empty:
        return {"win_rate": 0.0, "draw_rate": 0.0, "loss_rate": 0.0}
    
    try:
        total_win_rate = 0
        total_draw_rate = 0
        total_loss_rate = 0
        total_games = 0
        
        for _, row in win_rate_data.iterrows():
            if all(col in row for col in ['Win Rate (%)', 'Draw Rate (%)', 'Loss Rate (%)', 'Total Games']):
                games = row['Total Games']
                total_win_rate += row['Win Rate (%)'] * games
                total_draw_rate += row['Draw Rate (%)'] * games
                total_loss_rate += row['Loss Rate (%)'] * games
                total_games += games
        
        if total_games > 0:
            return {
                "win_rate": total_win_rate / total_games,
                "draw_rate": total_draw_rate / total_games,
                "loss_rate": total_loss_rate / total_games
            }
    except Exception as e:
        logger.error(f"Error calculating win stats: {e}")
    
    return {"win_rate": 0.0, "draw_rate": 0.0, "loss_rate": 0.0}

# Performance monitoring functions
def time_function(func):
    """Decorator to time function execution"""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Apply timing decorator to main processing functions for monitoring
fetch_and_preprocess = time_function(fetch_and_preprocess)
process_elo_progression = time_function(process_elo_progression)
process_win_rates = time_function(process_win_rates)
process_game_results = time_function(process_game_results)
process_opening_performance = time_function(process_opening_performance)