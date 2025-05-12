import os
import subprocess
from data_generation import main as generate_data

TEMP_DATA_FILE = "temp_data.csv"

def run_dashboard():
    """Run the Streamlit dashboard."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard_app.py")
    subprocess.run(["streamlit", "run", dashboard_path, "--", TEMP_DATA_FILE])

if __name__ == "__main__":
    # Step 1: Generate the data
    print("Running data generation...")
    games_df = generate_data()  # Get the DataFrame directly

    # Step 2: Save the data to a temporary file
    if games_df is not None:
        print(f"Saving data to {TEMP_DATA_FILE}...")
        games_df.to_csv(TEMP_DATA_FILE, index=False)

        # Step 3: Launch the dashboard
        print("Launching the dashboard...")
        run_dashboard()
    else:
        print("Data generation failed. Exiting.")