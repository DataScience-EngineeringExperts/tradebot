import os
import logging
from dotenv import load_dotenv
import pandas as pd
from pymongo import MongoClient
import alpaca_trade_api as tradeapi
from risk_strategy import RiskManagement, risk_params, send_teams_message, PortfolioManager
from trade_stats import record_trade

# Load environment variables
load_dotenv()

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')
TEAMS_WEBHOOK_URL = os.getenv('TEAMS_WEBHOOK_URL')

# Setup logging
logging.basicConfig(filename='../logfile.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Setup MongoDB connection
client = MongoClient(MONGO_CONN_STRING)
db = client.stock_data  # Use your actual database name
collection = db.crypto_data  # Use your actual collection name

def get_data_from_mongo():
    try:
        logging.info("Attempting to retrieve data from MongoDB...")
        documents = list(collection.find())
        if documents:
            logging.info(f"Retrieved {len(documents)} documents from MongoDB.")
        else:
            logging.info("No documents found in MongoDB collection.")
        data = pd.DataFrame(documents)
        logging.info(f"DataFrame shape after MongoDB retrieval: {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Error retrieving data from MongoDB: {e}")
        return pd.DataFrame()

def get_symbol(row):
    crypto = row["Crypto"]
    quote = row["Quote"]
    symbol = f"{crypto}{quote}"
    return symbol if symbol != 'nannan' else None

def calculate_SMA(data, window=5):
    return data['Close'].rolling(window=window).mean()

def calculate_EMA(data, window=5):
    return data['Close'].ewm(span=window, adjust=False).mean()

def calculate_RSI(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_MACD(data, short_window=12, long_window=26, signal_window=9):
    ema_short = calculate_EMA(data, window=short_window)
    ema_long = calculate_EMA(data, window=long_window)
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line, signal_line

def analyze_trend(short_term_SMA, long_term_SMA):
    return short_term_SMA > long_term_SMA

def process_buy(api, data, row, risk_management, teams_url, manager):
    symbol = get_symbol(row)
    if symbol is None:
        return

    row = data[data['Symbol'] == symbol].iloc[-1]
    short_term_SMA = calculate_SMA(data, window=5)
    long_term_SMA = calculate_SMA(data, window=10)
    rsi = calculate_RSI(data)
    macd_line, signal_line = calculate_MACD(data)

    upward_trend = analyze_trend(short_term_SMA.iloc[-1], long_term_SMA.iloc[-1])
    signal = row["Momentum Signal"]
    date = row["Date"]
    momentum_signal = row["Momentum Signal"]

    logging.info(f"Processing symbol: {symbol}, Signal: {signal}, Date Chose: {date}")
    print(f"Processing symbol: {symbol}, Signal: {signal}, Date Chose: {date}")

    risk_management.check_momentum(symbol, momentum_signal)

    if pd.isnull(signal) or signal != "Buy":
        return

    if upward_trend and rsi.iloc[-1] < 55 and macd_line.iloc[-1] > signal_line.iloc[-1]:
        logging.info(f"Buy conditions met for {symbol}")

        avg_entry_price = risk_management.get_avg_entry_price(symbol)
        print(f'Your average entry price was: {avg_entry_price}')
        entry_price = risk_management.get_current_price(symbol)

        logging.info(f"Average entry price for {symbol}: {avg_entry_price}")

        quantity = risk_management.calculate_quantity(symbol)
        logging.info(f"Calculated quantity to buy: {quantity}")

        if risk_management.validate_trade(symbol, quantity, "buy"):
            logging.info(f"Buy order validated for {symbol}")
            print(f"Buy order validated for {symbol}")

            if quantity > 0:
                order_details = {
                    'symbol': symbol,
                    'qty': quantity,
                    'side': 'buy',
                    'type': 'market',
                    'time_in_force': 'gtc'
                }

                try:
                    api.submit_order(**order_details)
                    logging.info(f'Buy order placed for {quantity} units of {symbol}.')
                    manager.add_asset(symbol, quantity, avg_entry_price * quantity)
                    manager.increment_operations()
                except Exception as e:
                    logging.error(f'Error placing buy order for {quantity} units of {symbol}: {str(e)}')
                    print(f'Error placing buy order for {quantity} units of {symbol}: {str(e)}')
                    return

                logging.info(f'Buy order placed for {quantity} units of {symbol}')
                send_teams_message(teams_url, {"text": f"Placed a BUY order for {quantity} units of {symbol}"})
                record_trade(symbol, 'buy', quantity, date)
            else:
                logging.info(f"Order quantity for symbol {symbol} is not greater than 0. Can't place the order.")
        else:
            logging.info(f"Buy order not validated for {symbol}")

def process_sell(api, data, row, risk_management, teams_url, manager):
    symbol = get_symbol(row)
    if symbol is None:
        return

    row = data[data['Symbol'] == symbol].iloc[-1]
    signal = row["Momentum Signal"]
    date = row["Date"]

    if pd.isnull(signal) or signal != "Sell":
        return

    try:
        # Retrieve all positions and find the one matching the symbol
        positions = api.list_positions()
        position = next((pos for pos in positions if pos.symbol == symbol), None)

        if position:
            quantity = float(position.qty) if isinstance(position.qty, str) else position.qty

            if quantity > 0:
                current_price = api.get_last_trade(symbol).price
                moving_avg = api.get_barset(symbol, 'day', limit=10).df[symbol]['close'].mean()

                if current_price > moving_avg:
                    quantity_to_sell = max(1, int(float(quantity) * 0.05))
                else:
                    quantity_to_sell = max(1, int(float(quantity) * 0.05))
            else:
                logging.info(f"Order quantity for symbol {symbol} is not greater than 0. Can't place the order.")
                quantity_to_sell = 0

            if quantity_to_sell > 0 and risk_management.validate_trade(symbol, quantity_to_sell, "sell"):
                try:
                    api.submit_order(
                        symbol=symbol,
                        qty=quantity_to_sell,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    manager.update_asset_value(symbol, (quantity - quantity_to_sell) * current_price)
                    manager.increment_operations()
                except Exception as e:
                    logging.error(f'Error placing sell order for {quantity_to_sell} units of {symbol}: {str(e)}')
                    return

                logging.info(f'Sell order placed for {quantity_to_sell} units of {symbol}')
                send_teams_message(teams_url, {"text": f"Placed a SELL order for {quantity_to_sell} units of {symbol}"})
                record_trade(symbol, 'sell', quantity_to_sell, date)
            else:
                logging.info(f"Sell order not validated for {symbol}")
        else:
            logging.info(f"No position found for symbol {symbol}")
    except Exception as e:
        logging.error(f'Error getting position or placing sell order for {symbol}: {str(e)}')

def process_signals():
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets')
    manager = PortfolioManager(api)
    risk_management = RiskManagement(api, risk_params)
    data = get_data_from_mongo()

    if data.empty:
        logging.info("No data retrieved from MongoDB.")
        return

    data['Symbol'] = data.apply(get_symbol, axis=1)
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by='Date', ascending=True, inplace=True)
    grouped = data.sort_values('Date').groupby('Symbol').tail(1)

    for index, row in grouped.iterrows():
        process_buy(api, data, row, risk_management, TEAMS_WEBHOOK_URL, manager)
        process_sell(api, data, row, risk_management, TEAMS_WEBHOOK_URL, manager)

if __name__ == "__main__":
    print('Script started')
    process_signals()
    print('Script ended')
