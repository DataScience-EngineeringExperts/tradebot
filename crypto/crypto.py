import requests
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import concurrent.futures
import logging
import pandas as pd

# Setup logging
logging.basicConfig(filename='../logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')
ALPHA_VANTAGE_API = os.getenv('ALPHA_VANTAGE_API')

# Connect to MongoDB
client = MongoClient(MONGO_CONN_STRING)
db = client.stock_data  # Assuming 'stock_data' is the database name
collection = db.crypto_data  # Assuming 'crypto_data' is the collection name

# Define the pairs
# Supported cryptocurrencies
supported_cryptos = ['AAVE', 'AVAX', 'BAT', 'BCH', 'BTC', 'CRV', 'DOT', 'ETH', 'GRT', 'LINK', 'LTC', 'MKR', 'SHIB', 'UNI', 'USDC', 'USDT', 'XTZ']

# Define pairings
btc_pairs = ['BCH', 'ETH', 'LTC', 'UNI']
usdt_pairs = ['AAVE', 'BCH', 'BTC', 'ETH', 'LINK', 'LTC', 'UNI']
usdc_pairs = ['AAVE', 'AVAX', 'BAT', 'BCH', 'BTC', 'CRV', 'DOT', 'ETH', 'GRT', 'LINK', 'LTC', 'MKR', 'SHIB', 'UNI', 'XTZ']
usd_pairs = ['AAVE', 'AVAX', 'BAT', 'BCH', 'BTC', 'CRV', 'DOT', 'ETH', 'GRT', 'LINK', 'LTC', 'MKR', 'SHIB', 'UNI', 'USDC', 'USDT', 'XTZ']

# Combine base and quote currencies to form tradeable pairs
tradeable_btc_pairs = [f"{pair}/BTC" for pair in btc_pairs if pair in supported_cryptos]
tradeable_usdt_pairs = [f"{pair}/USDT" for pair in usdt_pairs if pair in supported_cryptos]
tradeable_usdc_pairs = [f"{pair}/USDC" for pair in usdc_pairs if pair in supported_cryptos]
tradeable_usd_pairs = [f"{pair}/USD" for pair in usd_pairs if pair in supported_cryptos]

# Combine all pairs
all_tradeable_pairs = tradeable_btc_pairs + tradeable_usdt_pairs + tradeable_usdc_pairs + tradeable_usd_pairs

# Remove duplicates (if any)
all_tradeable_pairs = list(set(all_tradeable_pairs))

# Sort for better readability
all_tradeable_pairs.sort()

# Print all tradeable pairs
print(f"Total Tradeable Pairs: {len(all_tradeable_pairs)}")
print(all_tradeable_pairs)

all_pairs = all_tradeable_pairs

# Create a session object
s = requests.Session()

def fetch_exchange_rate(base_currency, quote_currency):
    url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={base_currency}&to_currency={quote_currency}&apikey={ALPHA_VANTAGE_API}"
    try:
        response = s.get(url).json()
        # print(f'here is the response: {response}')
        exchange_rate = response["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
        # print(f'here is the exchange rate if we found it: {exchange_rate}')
        return float(exchange_rate)
    except KeyError:
        #logging.error(f"No exchange rate found from {base_currency} to {quote_currency}")
        return None
    except Exception as e:
        # logging.error(f"Error while fetching exchange rate: {e}")
        return None


def store_in_mongodb(data):
    try:
        # Delete all existing records in the collection
        collection.delete_many({})  # This matches and deletes all documents

        # Insert new data into the MongoDB collection
        if not data.empty:  # Check if the DataFrame is not empty to avoid insertion errors
            collection.insert_many(data.to_dict('records'))
            # logging.info("Existing records deleted and new data stored in MongoDB successfully")
        else:
            logging.info("No new data to store in MongoDB")
    except Exception as e:
        logging.error(f"Error while updating data in MongoDB: {e}")


def build_dataframe(latest_intraday_data, exchange_rate, base_crypto, quote):
    dates, opens, highs, lows, closes, volumes = [], [], [], [], [], []
    for date, data in latest_intraday_data:
        dates.append(date)
        opens.append(round(float(data['1. open']) * exchange_rate, 2))
        highs.append(round(float(data['2. high']) * exchange_rate, 2))
        lows.append(round(float(data['3. low']) * exchange_rate, 2))
        closes.append(round(float(data['4. close']) * exchange_rate, 2))
        volumes.append(round(float(data['5. volume']) * exchange_rate, 2))

    df = pd.DataFrame({
        'Date': dates,
        'Crypto': [base_crypto] * len(latest_intraday_data),
        'Quote': [quote] * len(latest_intraday_data),
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes,
    })

    return df.sort_values('Date')


def fetch_and_process_data(crypto):
    base_crypto, quote = crypto.split('/')
    exchange_rate = fetch_exchange_rate(base_crypto, quote)

    if exchange_rate is None:
        # logging.error(f"No exchange rate found for {base_crypto} to {quote}")
        return

    url = f"https://www.alphavantage.co/query?function=CRYPTO_INTRADAY&symbol={base_crypto}&market=USD&interval=5min&outputsize=full&apikey={ALPHA_VANTAGE_API}"
    try:
        response = s.get(url).json()
        intraday_data = response.get('Time Series Crypto (5min)', {})
        if not intraday_data:
            logging.error(f"No intraday data found for {crypto}")
            return

        latest_intraday_data = list(intraday_data.items())[:288]  # Get the latest 288 5-minute intervals
        df = build_dataframe(latest_intraday_data, exchange_rate, base_crypto, quote)
        processed_data = apply_strategies(df)
        store_in_mongodb(processed_data)
    except Exception as e:
        logging.error(f"Error while fetching intraday stats: {e}")


def apply_strategies(df):
    window_size = 20
    std_dev_factor = 1
    period = 14

    df['Mean'] = df['Close'].rolling(window=window_size).mean()
    df['Std Dev'] = df['Close'].rolling(window=window_size).std()
    df['Buy Signal'] = df['Close'] < (df['Mean'] - std_dev_factor * df['Std Dev'])
    df['Sell Signal'] = df['Close'] > (df['Mean'] + std_dev_factor * df['Std Dev'])
    df['Mean Reversion Signal'] = 'Hold'
    df.loc[df['Buy Signal'], 'Mean Reversion Signal'] = 'Buy'
    df.loc[df['Sell Signal'], 'Mean Reversion Signal'] = 'Sell'

    df['Momentum'] = df['Close'].diff(period)
    df['Momentum Signal'] = 'Hold'
    df.loc[df['Momentum'] > 0, 'Momentum Signal'] = 'Buy'
    df.loc[df['Momentum'] < 0, 'Momentum Signal'] = 'Sell'

    return df


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(fetch_and_process_data, pair) for pair in all_pairs]
    for future in concurrent.futures.as_completed(futures):
        future.result()