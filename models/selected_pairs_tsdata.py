import os
from time import sleep
import requests
import pymongo
from dotenv import load_dotenv
from datetime import datetime
import logging
import time
import concurrent.futures


# Setup logging
logging.basicConfig(filename='../logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API')
MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

FUNCTIONS = [
    "TIME_SERIES_INTRADAY",
    "TIME_SERIES_DAILY",
    "TIME_SERIES_DAILY_ADJUSTED"
]


def fetch_unique_symbols(mongo_conn_string):
    client = pymongo.MongoClient(mongo_conn_string)
    db = client['stock_data']  # Specify your database name
    collection = db['selected_pairs']  # Specify your collection name

    # Fetch unique symbols using the distinct method
    unique_symbols = collection.distinct('symbol')

    client.close()  # Don't forget to close the connection
    return unique_symbols


def fetch_data(symbol, api_key):
    data_document = {'symbol': symbol, 'datetime_imported': datetime.now()}

    BASE_URL = "https://www.alphavantage.co/query"

    for function in FUNCTIONS:
        url = f"{BASE_URL}?function={function}&symbol={symbol}&apikey={api_key}"

        # Append interval and outputsize for intraday data
        if function == "TIME_SERIES_INTRADAY":
            url += "&interval=60min&outputsize=full"
        # Extend outputsize for daily and adjusted daily data
        elif function in ["TIME_SERIES_DAILY", "TIME_SERIES_DAILY_ADJUSTED"]:
            url += "&outputsize=full"  # Request full historical data

        r = requests.get(url)
        if r.status_code != 200:
            logging.error(f"HTTP Error for {symbol} with {function}: Status code {r.status_code} - {r.text}")
            continue  # Continue to the next function if current one fails

        data = r.json()

        # Determine the key for time series data based on the function
        time_series_key = None
        if function in ["TIME_SERIES_INTRADAY", "TIME_SERIES_DAILY", "TIME_SERIES_DAILY_ADJUSTED"]:
            time_series_key = [key for key in data if 'Time Series' in key or 'Global Quote' in key][0]

        if time_series_key:
            time_series_data = data.get(time_series_key, {})

            closing_prices = []
            for timestamp, values in time_series_data.items():
                # Handle data structure for 'Global Quote'
                if function == 'GLOBAL_QUOTE':
                    closing_price = values.get('05. price')
                else:
                    closing_price = values.get('4. close')

                if closing_price:
                    closing_prices.append({'timestamp': timestamp, 'close_price': closing_price})

            data_document[f'{function}_data'] = closing_prices
        else:
            logging.error(f"Failed to fetch or parse data for {symbol} with {function}. Data might be missing the expected key.")
            data_document[f'{function}_data'] = None

        sleep(12)  # Sleep to avoid hitting the API rate limit

    return data_document


def store_data_in_mongo(client, data_document):
    db = client['stock_data']  # Database name
    collection = db['aggregated_stock_data']  # Collection name

    for function, new_data in data_document.items():
        if function.endswith('_data') and new_data:  # Process only time series data
            # Find existing document for the symbol
            existing_document = collection.find_one({'symbol': data_document['symbol']})

            if existing_document:
                # For each new data point, check if it exists in the existing document and append if new
                for new_entry in new_data:
                    new_timestamp = new_entry['timestamp']
                    # Convert string timestamp to datetime object for comparison
                    new_timestamp_dt = datetime.strptime(new_timestamp, "%Y-%m-%d" if len(new_timestamp) == 10 else "%Y-%m-%d %H:%M:%S")

                    # Check if this timestamp exists in the existing data
                    existing_entry = next((item for item in existing_document.get(function, []) if item['timestamp'] == new_timestamp), None)

                    if not existing_entry:
                        # Append new data point if it doesn't exist
                        collection.update_one({'_id': existing_document['_id']}, {'$push': {function: new_entry}})
                        #logging.info(f"Appended new data for {data_document['symbol']} on {new_timestamp} to existing document.")
            else:
                # If no existing document is found, create a new one with the current data batch
                new_doc = {
                    'symbol': data_document['symbol'],
                    'datetime_imported': data_document['datetime_imported'],
                    function: new_data
                }
                collection.insert_one(new_doc)
                logging.info(f"Inserted new document for {data_document['symbol']} with data for {function}.")

            sleep(1)  # Respectful sleep to avoid hitting API or database operation limits too quickly


# Connect to MongoDB
mongo_client = pymongo.MongoClient(MONGO_CONN_STRING)

unique_symbols = fetch_unique_symbols(MONGO_CONN_STRING)


def process_ticker(ticker):
    try:
        logging.info(f"Processing data for {ticker}...")
        data_document = fetch_data(ticker, ALPHA_VANTAGE_API_KEY)
        store_data_in_mongo(mongo_client, data_document)
    except Exception as e:
        logging.error(f"An error occurred during processing {ticker}: {e}")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting to fetch unique symbols...")
    unique_symbols = fetch_unique_symbols(MONGO_CONN_STRING)
    logging.info(f"Fetched {len(unique_symbols)} unique symbols.")

    mongo_client = pymongo.MongoClient(MONGO_CONN_STRING)
    logging.info("MongoDB client established.")

    # Set up a counter and time tracker
    api_calls_made = 0
    start_time = time.time()

    logging.info("Beginning processing of ticker symbols...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for ticker in unique_symbols:
            if api_calls_made >= 150:
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                if elapsed_time < 60:
                    sleep_time = 60 - elapsed_time
                    logging.info(f"API limit reached, taking a break for {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)  # Pause execution to respect API limit

                # Reset counter and timer
                api_calls_made = 0
                start_time = time.time()

            futures.append(executor.submit(process_ticker, ticker))
            api_calls_made += 1  # Increment counter for each API call made

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

    mongo_client.close()
    logging.info("Data fetching and storage process completed.")

