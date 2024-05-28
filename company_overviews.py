import sys
import time
import os
import concurrent.futures
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from sector_filters import get_sector_thresholds, get_market_condition_adjustments
from mongo_func import MongoManager  # mongo_func.py class module for interacting with mongo

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("logfile.log"), logging.StreamHandler(sys.stdout)])

# Load environment variables
load_dotenv()
API_KEY = os.getenv('ALPHA_VANTAGE_API')
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Initialize the MongoManager
mongo_manager = MongoManager(MONGO_DB_CONN_STRING, 'stock_data')

# Then, within your main script or where you're applying the filters:
sector_thresholds = get_sector_thresholds()
market_condition_adjustments = get_market_condition_adjustments()

def adjust_filters_for_market_conditions(filters):
    # Adjust filter thresholds based on current market conditions
    adjusted_filters = filters.copy()
    for key, adjustment in market_condition_adjustments.items():
        if key in adjusted_filters:
            if isinstance(adjusted_filters[key], list) and len(adjusted_filters[key]) == 2:  # Assuming it's a range
                adjusted_filters[key] = [adjusted_filters[key][0] + adjustment, adjusted_filters[key][1] + adjustment]
            else:
                adjusted_filters[key] += adjustment
    return adjusted_filters


def safe_float(value, default=0.0):
    """Safely convert a value to float, returning a default if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def retrieve_company_overview(api_key, symbol):
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if 'Sector' not in data or data['Sector'] in [None, 'None', '', ' ', 'N/A']:
            # logging.info(f"Sector information is missing for {symbol}. Skipping.")
            print(f'{symbol} data is: {data}')
            return None

        sector = data['Sector']
        filters = sector_thresholds.get(sector, sector_thresholds['default'])
        filters = adjust_filters_for_market_conditions(filters)

        # Convert and check all necessary fields against the filters
        if not all([
            safe_float(data.get('MarketCapitalization')) >= filters['MarketCapitalization'],
            safe_float(data.get('EBITDA')) >= filters['EBITDA'],
            filters['PERatio'][0] <= safe_float(data.get('PERatio'), float('inf')) <= filters['PERatio'][1],
            safe_float(data.get('EPS')) >= filters['EPS'],
            safe_float(data.get('Beta')) <= filters['Beta']
        ]):
            # logging.error(f"Symbol {symbol} does not meet filter criteria. Skipping.")
            return None

        data['date_added'] = datetime.now()

        return data

    except requests.exceptions.RequestException as e:
        logging.error(f"Error retrieving company overview for symbol '{symbol}': {e}")
        return None


def fetch_and_store_company_overviews(api_key, tickers_list, mongo_manager, collection_name):
    RATE_LIMIT_CALLS = 285  # Adjust the API call limit here
    company_overviews_list = []
    processed_requests = 0
    total_tickers = len(tickers_list)
    start_time = datetime.now()
    request_timestamps = []  # List to keep track of request timestamps

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_ticker = {executor.submit(retrieve_company_overview, api_key, ticker): ticker for ticker in tickers_list}
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            data = future.result()

            # Check if we've reached the rate limit
            if len(request_timestamps) >= RATE_LIMIT_CALLS:
                # Calculate time since the RATE_LIMIT_CALLS-th last request
                time_since_limit_last_request = datetime.now() - request_timestamps[-RATE_LIMIT_CALLS]
                # If it's been less than a minute, sleep for the remaining time
                if time_since_limit_last_request.total_seconds() < 60:
                    sleep_time = 60 - time_since_limit_last_request.total_seconds()
                    logging.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)

            request_timestamps.append(datetime.now())  # Record the timestamp of the request
            # Remove timestamps older than a minute to keep the list size manageable
            request_timestamps = [ts for ts in request_timestamps if datetime.now() - ts <= timedelta(minutes=1)]

            if data:
                company_overviews_list.append(data)
                processed_requests += 1

            # Logging progress and time remaining after processing each future
            progress = (processed_requests / total_tickers) * 100
            elapsed_time = (datetime.now() - start_time).total_seconds()
            time_per_request = elapsed_time / max(processed_requests, 1)  # Avoid division by zero
            time_remaining = (total_tickers - processed_requests) * time_per_request
            logging.info(f"Progress: {progress:.2f}%, Time remaining: {time_remaining:.2f} seconds")

    # Use MongoManager to insert documents with deduplication
    mongo_manager.insert_with_deduplication(collection_name, company_overviews_list)


def main():
    start_time = datetime.now()

    # MongoDB collection name
    collection_name = 'company_overviews'

    tickers_collection = mongo_manager.db['tickers']
    tickers_list = [ticker['symbol'] for ticker in tickers_collection.find({}, {'symbol': 1, '_id': 0})]

   #tickers_list = tickers_list[:25]

    # Pass mongo_manager and collection_name to the function
    fetch_and_store_company_overviews(API_KEY, tickers_list, mongo_manager, collection_name)

    total_time = (datetime.now() - start_time).total_seconds()
    logging.info(f"Process completed. Total elapsed time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
