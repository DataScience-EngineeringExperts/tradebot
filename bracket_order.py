from risk_strategy import RiskManagement, risk_params
from alpha_vantage.timeseries import TimeSeries
import alpaca_trade_api as tradeapi
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Database configuration
MONGO_DB_CONN_STRING = os.getenv("MONGO_CONN_STRING")

# Initialize MongoDB client
mongo_client = MongoClient(MONGO_DB_CONN_STRING)

# Select the database
db = mongo_client["stock_data"]  # Use your actual database name here

# Select the collection
selected_pairs_collection = db['selected_pairs']

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

ALPHA_VANTAGE_API = os.getenv("ALPHA_VANTAGE_API")

# Alpaca and Alpha Vantage API setup
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets')
ts = TimeSeries(key=ALPHA_VANTAGE_API)
rm = RiskManagement(api, risk_params)

# Your Microsoft Teams channel webhook URL
teams_url = os.getenv('TEAMS_WEBHOOK_URL')

def get_symbols_from_mongodb():
    try:
        symbols = selected_pairs_collection.find({}, {'_id': 0, 'Symbol': 1})
        return [doc['Symbol'] for doc in symbols]
    except Exception as e:
        logging.error(f"Error fetching symbols from MongoDB: {e}")
        return []

def get_holdings(api):
    try:
        current_positions = api.list_positions()
        return {position.symbol: position.qty for position in current_positions}
    except Exception as e:
        logging.error(f"Error fetching current holdings: {e}")
        return {}

def send_teams_message(message):
    message = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": "0076D7",
        "summary": "Trade Orders Summary",
        "sections": [{
            "activityTitle": "Trade Orders Placed",
            "activitySubtitle": "Summary of Buy and Sell Orders",
            "facts": [{
                "name": "Orders",
                "value": message
            }],
            "markdown": True
        }]
    }
    headers = {
        "Content-type": "application/json",
    }
    response = requests.post(teams_url, headers=headers, data=json.dumps(message))
    return response.status_code


def place_order(api, symbol, shares, close_price):
    try:
        # Check if the drawdown limit has been reached
        if rm.check_risk_before_order():
            take_profit = {"limit_price": round(close_price * 1.0243, 2)}
            stop_loss = {"stop_price": round(close_price * 0.9821, 2)}
            client_order_id = f"gcos_{random.randrange(100000000)}"
            print(f"{symbol}: Attempting to place an order!")

            order = api.submit_order(
                symbol=symbol,
                qty=round(float(shares)),  # Shares rounded to the nearest whole number
                side='buy',
                type='limit',
                limit_price=round(close_price, 2),
                order_class='bracket',
                take_profit=take_profit,
                stop_loss=stop_loss,
                client_order_id=client_order_id,
                time_in_force='day'
            )
            print(f"{symbol}: order placed successfully!")

            # # Print the response from the API
            # print(f"{symbol}: Order API response: {order}")

            # Create a message to send to Teams channel
            message = f"Order placed successfully! Symbol: {symbol}, Shares: {shares}, Price: {close_price}"
            # Send message to Teams
            send_teams_message(message)

            return True
        else:
            print(f"Max drawdown limit reached, order for {symbol} not placed")

    except Exception as e:
        print(f"Order for {symbol} could not be placed: {str(e)}")
        return False

cash_balance = api.get_account().cash
portfolio_balance = float(api.get_account().portfolio_value)
maximum_risk_per_trade = rm.risk_params['max_risk_per_trade']


def get_open_orders(api):
    open_orders = api.list_orders(status='open')
    open_orders_symbols = [order.symbol for order in open_orders]
    return open_orders_symbols


def handle_symbol(symbol):
    try:
        # Get the current holdings and open orders before checking the conditions
        current_holdings = get_holdings(api)
        open_orders_symbols = get_open_orders(api)

        # Prepare API URLs
        daily_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API}'
        rsi_url = f'https://www.alphavantage.co/query?function=RSI&symbol={symbol}&interval=daily&time_period=14&series_type=close&apikey={ALPHA_VANTAGE_API}'
        macd_url = f'https://www.alphavantage.co/query?function=MACD&symbol={symbol}&interval=daily&series_type=close&apikey={ALPHA_VANTAGE_API}'
        sma_url = f'https://www.alphavantage.co/query?function=SMA&symbol={symbol}&interval=daily&time_period=30&series_type=close&apikey={ALPHA_VANTAGE_API}'

        # Make API requests
        daily_data = requests.get(daily_url).json()
        rsi_data = requests.get(rsi_url).json()
        macd_data = requests.get(macd_url).json()
        sma_data = requests.get(sma_url).json()

        # Extract the first data point for each technical indicator
        daily_point = list(daily_data['Time Series (Daily)'].values())[0]
        rsi_point = list(rsi_data['Technical Analysis: RSI'].values())[0]
        macd_point = list(macd_data['Technical Analysis: MACD'].values())[0]
        sma_point = list(sma_data['Technical Analysis: SMA'].values())[0]

        recent_close = float(daily_point['4. close'])
        recent_rsi = float(rsi_point['RSI'])
        recent_macd = float(macd_point['MACD'])
        recent_signal = float(macd_point['MACD_Signal'])
        recent_sma = float(sma_point['SMA'])

        if recent_rsi <= 30:
            print(f"{symbol}: RSI condition met.")
        else:
            print(f"{symbol}: RSI condition not met.")

        if recent_macd >= recent_signal:
            print(f"{symbol}: MACD condition met.")
        else:
            print(f"{symbol}: MACD condition not met.")

        if recent_close >= recent_sma:
            print(f"{symbol}: Price above SMA condition met.")
        else:
            print(f"{symbol}: Price above SMA condition not met.")

        if recent_rsi <= 30 and recent_macd >= recent_signal and recent_close >= recent_sma:
            # Check if we already have a position or an open order for this symbol
            if symbol in current_holdings or symbol in open_orders_symbols:
                print(f"Already hold a position or have an open order in {symbol}, skipping order...")
            else:
                shares = int(portfolio_balance * maximum_risk_per_trade) / recent_close

                print(f"{symbol}: All conditions met. Place order for: {shares} shares.")

                place_order(api, symbol, shares, recent_close)
        else:
            print(f"{symbol}: Not all conditions met. No order placed.")
    except ValueError:
        print(f"Unable to fetch data for {symbol}. Skipping...")
    except Exception as e:
        print(f"An unexpected error occurred for {symbol}: {str(e)}")


if __name__ == "__main__":
    symbols = get_symbols_from_mongodb()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(handle_symbol, symbol) for symbol in symbols]

    for future in as_completed(futures):
        try:
            data = future.result()
        except Exception as exc:
            logging.error(f"An exception occurred in a thread: {exc}")
