import subprocess
import sys
import time
import os
import logging

# Create a logger
logging.basicConfig(filename='logfile.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Get the current script's absolute directory path
script_dir = os.path.dirname(os.path.abspath(__file__))

# List your scripts in the desired order of execution
scripts = [
    "alphavantagetickers.py",
    "company_overviews.py",
    "selected_pairs_history.py",
    "bracket_order.py"
]

# Function to run the script and handle errors with retries
def run_script_with_retries(script, max_retries=2, wait_time=20):
    retries = 0
    while retries <= max_retries:
        logging.info(f"Running {script}")

        # Construct the absolute path of the script
        script_path = os.path.join(script_dir, script)

        # Run the script and capture its exit code
        try:
            subprocess.check_call(["python3", script_path])
            break  # If the script is successful, break the loop
        except subprocess.CalledProcessError as e:
            logging.error(f"Error occurred while running {script}, exit code: {e.returncode}")
            retries += 1
            if retries <= max_retries:
                logging.info(f"Retrying {script} after waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"Script {script} failed after {max_retries} retries. Skipping this script and moving to the next one.")
                break

# Iterate over the scripts list and execute them one by one
for script in scripts:
    run_script_with_retries(script)
