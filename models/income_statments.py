import os
import requests
from dotenv import load_dotenv
import logging
from pymongo import MongoClient
import concurrent.futures
import time
from datetime import datetime, timezone


# Setup logging
logging.basicConfig(filename='../logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API')
MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Setup MongoDB connection
client = MongoClient(MONGO_CONN_STRING)
db = client['stock_data']  # Accessing the 'stock_data' database

# Accessing the 'selected_pairs' collection to fetch symbols
selected_pairs_collection = db['selected_pairs']

# Global variable to count API calls
api_calls_count = 0

# Function to fetch income statement
def fetch_income_statement(symbol):
    global api_calls_count

    # Rate limiting: 75 calls per minute
    if api_calls_count >= 75:
        logging.info("API call limit reached, waiting...")
        time.sleep(60)  # Wait for 60 seconds before proceeding
        api_calls_count = 0  # Reset the counter

    url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    api_calls_count += 1  # Increment the API call counter

    if response.status_code == 200:
        data = response.json()
        logging.info(f"Successfully fetched income statement for {symbol}")
        return data
    else:
        logging.error(f"Failed to fetch income statement for {symbol} with status code {response.status_code}")
        return None

def process_symbol(document):
    symbol = document.get('symbol')
    if symbol:
        income_statement_data = fetch_income_statement(symbol)
        if income_statement_data:
            #print(f"Data for {symbol}:", income_statement_data)
            # Access the 'technicals' collection
            technicals_collection = db['technicals']
            # Prepare the document to insert or update
            document_to_upsert = {
                'symbol': symbol,
                'income_statement': income_statement_data,
                'datetime_imported': datetime.now(timezone.utc)  # Set the current datetime in UTC
            }
            # Update the document with the given symbol, or insert it if it doesn't exist
            technicals_collection.update_one(
                {'symbol': symbol},  # Filter by symbol
                {'$set': document_to_upsert},  # Update or set fields
                upsert=True  # Insert the document if it doesn't exist
            )
        else:
            print(f"Failed to fetch data for {symbol}")

# Fetch symbols from MongoDB 'selected_pairs' collection
symbols = list(selected_pairs_collection.find())

# Using ThreadPoolExecutor to manage multiple threads
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Submit tasks to the executor
    futures = [executor.submit(process_symbol, symbol) for symbol in symbols]

    # Wait for all futures to complete
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as exc:
            logging.error(f'Generated an exception: {exc}')
