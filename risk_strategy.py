import alpaca_trade_api as tradeapi
import requests
import json
from trade_stats import download_trades
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from alpaca_trade_api.rest import REST, APIError
from datetime import datetime, timedelta
from port_op import optimize_portfolio
import os
from dotenv import load_dotenv
import logging
import pandas as pd
import numpy as np
from scipy.stats import norm
from pymongo import MongoClient

# Setup logging
logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Alpaca credentials
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # or https://api.alpaca.markets for live trading
ALPHA_VANTAGE_API = os.getenv('ALPHA_VANTAGE_API')


# Connect to MongoDB
client = MongoClient(MONGO_CONN_STRING)

alpha_vantage_ts = TimeSeries(key=ALPHA_VANTAGE_API, output_format='pandas')
alpha_vantage_crypto = CryptoCurrencies(key=ALPHA_VANTAGE_API, output_format='pandas')

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)


account = api.get_account()
equity = float(account.equity)


def load_risk_params():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    risk_params_file_path = os.path.join(current_script_dir, 'risk_params.json')

    if not os.path.exists(risk_params_file_path):
        risk_params_file_path = os.path.join(current_script_dir, '..', 'risk_params.json')
        risk_params_file_path = os.path.normpath(risk_params_file_path)

    if os.path.exists(risk_params_file_path):
        with open(risk_params_file_path, 'r') as f:
            risk_params = json.load(f)
            # print("Risk parameters loaded:", risk_params)
            # logging.info("Risk parameters loaded successfully.")
            return risk_params
    else:
        logging.error(f"Error: 'risk_params.json' not found at {risk_params_file_path}")
        return None


risk_params = load_risk_params()
if risk_params:
    print("Risk parameters loaded successfully.")
    # Continue with your script using the loaded risk_params
else:
    print("Failed to load risk parameters.")


print(risk_params['max_position_size'])


class CryptoAsset:
    def __init__(self, symbol, quantity, value_usd):
        self.symbol = symbol
        self.quantity = quantity
        self.value_usd = value_usd
        self.value_24h_ago = None  # To store the value 24 hours ago

        # Connect to MongoDB and retrieve unique crypto pairs
        self.crypto_symbols = self.get_unique_crypto_pairs()

    def get_unique_crypto_pairs(self):
        # Connect to the MongoDB database
        db = client.stock_data
        collection = db.crypto_data

        # Retrieve unique pairs
        pipeline = [
            {"$group": {"_id": {"Crypto": "$Crypto", "Quote": "$Quote"}}},
            {"$project": {"_id": 0, "pair": {"$concat": ["$_id.Crypto", "/", "$_id.Quote"]}}}
        ]
        results = collection.aggregate(pipeline)
        unique_pairs = [doc['pair'] for doc in results]

        return unique_pairs

    def profit_loss_24h(self):
        if self.value_24h_ago is not None:
            return (self.value_usd - self.value_24h_ago) / self.value_24h_ago * 100
        else:
            return None


# Define a class to manage the portfolio
class PortfolioManager:
    def __init__(self, api):
        self.api = api
        self.assets = {}
        self.operations = 0  # track the number of operations

    def increment_operations(self):
        self.operations += 1

    def add_asset(self, symbol, quantity, value_usd):
        self.assets[symbol] = CryptoAsset(symbol, quantity, value_usd)

    def update_asset_value(self, symbol, value_usd):
        if symbol in self.assets:
            self.assets[symbol].value_usd = value_usd

    def portfolio_value(self):
        return sum(asset.value_usd for asset in self.assets.values())

    def portfolio_balance(self):
        return {symbol: (asset.value_usd / self.portfolio_value()) * 100 for symbol, asset in self.assets.items()}

    def sell_decision(self, symbol):
        balance = self.portfolio_balance()

        if balance[symbol] > 25 or balance[symbol] > 0.4 * sum(balance.values()):
            return True
        else:
            return False

    def scale_out(self, symbol):
        quantity_to_sell = int(self.assets[symbol].quantity * 0.1)  # Sell 10% of holdings
        return quantity_to_sell

    def update_asset_values_24h(self):
        for asset in self.assets.values():
            asset.value_24h_ago = asset.value_usd



### Usable functions for the RiskManagement class below
def get_exchange_rate(base_currency, quote_currency):
    # Your Alpha Vantage API key
    api_key = ALPHA_VANTAGE_API

    # Prepare the URL
    url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={base_currency}&to_currency={quote_currency}&apikey={api_key}"

    # Send GET request
    response = requests.get(url)

    # Parse JSON response
    data = json.loads(response.text)

    # Extract exchange rate
    exchange_rate = data["Realtime Currency Exchange Rate"]["5. Exchange Rate"]

    return float(exchange_rate)


def fetch_account_details():
    url = "https://paper-api.alpaca.markets/v2/account"
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Failed to fetch account details: {response.text}")
        return None


def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_greeks(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return {
        'delta': norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1),
        'gamma': norm.pdf(d1) / (S * sigma * np.sqrt(T)),
        'theta': -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2) if option_type == "call" else -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2),
        'vega': S * norm.pdf(d1) * np.sqrt(T)
    }

## RiskManagement class developed for usage on the account
class RiskManagement:
    crypto_symbols = ['AAVE/USD', 'ALGO/USD', 'AVAX/USD', 'BCH/USD', 'BTC/USD', 'ETH/USD',
                  'LINK/USD', 'LTC/USD', 'TRX/USD', 'UNI/USD', 'USDT/USD', 'SHIB/USD']
    def __init__(self, api, risk_params):
        self.api = api
        self.risk_params = risk_params
        self.alpha_vantage_crypto = CryptoCurrencies(key=ALPHA_VANTAGE_API, output_format='pandas')
        self.manager = PortfolioManager(api)
        self.crypto_value = 0
        self.commodity_value = 0
        self.crypto_value = 0
        self.commodity_value = 0
        self.options_crypto_notional_value = 0
        self.options_commodity_notional_value = 0
        self.max_options_allocation = 0.2  # Maximum 20% of portfolio allocated to options
        self.max_options_loss_threshold = 0.1

        self.crypto_symbols = ['AAVE/USD', 'ALGO/USD', 'AVAX/USD', 'BCH/USD', 'BTC/USD', 'ETH/USD',
                               'LINK/USD', 'LTC/USD', 'TRX/USD', 'UNI/USD', 'USDT/USD', 'SHIB/USD']
        self.TARGET_ALLOCATION = {
            'options': 0.20,  # 20%
            'crypto': 0.30,  # 30%
            'equities': 0.50  # 50%
        }
        self.initialize_account_info()

    def initialize_account_info(self):
        account = self.api.get_account()
        self.peak_portfolio_value = float(account.cash)

    def calculate_current_allocation(self):
        positions = self.api.list_positions()
        total_value = sum(float(position.market_value) for position in positions)
        allocation = {
            'options': 0,
            'crypto': 0,
            'equities': 0
        }

    def calculate_options_allocation(self):
        positions = self.api.list_positions()
        options_value = sum(float(position.market_value) for position in positions if 'OPT' in position.symbol)
        account_value = float(self.api.get_account().portfolio_value)
        return options_value / account_value

    def calculate_portfolio_value(self):
        positions = self.api.list_positions()
        portfolio_value = sum(float(position.market_value) for position in positions)
        return portfolio_value

    def calculate_current_allocation(self):
        positions = self.api.list_positions()
        total_value = sum(float(position.market_value) for position in positions)
        allocation = {
            'options': 0,
            'crypto': 0,
            'equities': 0
        }

        for position in positions:
            symbol = position.symbol
            market_value = float(position.market_value)
            if 'OPT' in symbol:
                allocation['options'] += market_value
            elif symbol.endswith('USD'):
                allocation['crypto'] += market_value
            else:
                allocation['equities'] += market_value

        for asset_class in allocation:
            if total_value != 0:
                allocation[asset_class] /= total_value

        return allocation

    def rebalance_portfolio(self):
        current_allocation = self.calculate_current_allocation()
        account = self.api.get_account()
        total_value = float(account.portfolio_value)

        for asset_class, target_pct in self.TARGET_ALLOCATION.items():
            current_pct = current_allocation[asset_class]
            diff_pct = target_pct - current_pct
            amount_to_trade = diff_pct * total_value

            if amount_to_trade > 0:
                self.buy_asset_class(asset_class, amount_to_trade)
            elif amount_to_trade < 0:
                self.sell_asset_class(asset_class, -amount_to_trade)

    def buy_asset_class(self, asset_class, amount_to_trade):
        account_details = fetch_account_details()
        if not account_details:
            return

        available_cash = float(account_details['cash'])

        if asset_class == 'crypto':
            for symbol in self.crypto_symbols:
                current_price = self.get_current_price(symbol)
                if current_price:
                    qty = self.calculate_quantity(symbol)
                    print(f'Amount to Trade: {amount_to_trade} and suggest calc qty was: {qty}')
                    if self.validate_trade(symbol, 'buy'):
                        for attempt in range(3):  # Retry mechanism
                            try:
                                self.api.submit_order(
                                    symbol=symbol,
                                    qty=qty,
                                    side='buy',
                                    type='market',
                                    time_in_force='gtc'
                                )
                                logging.info(f"Bought {qty} of {symbol} to rebalance crypto allocation.")
                                break
                            except tradeapi.rest.APIError as e:
                                if 'asset is not active' in str(e):
                                    logging.warning(f"Skipping {symbol} because it is not active.")
                                    break
                                elif 'insufficient balance' in str(e):
                                    logging.warning(f"Retrying {symbol} due to insufficient balance.")
                                    continue
                                else:
                                    logging.error(f"Failed to submit order for {symbol}: {e}")
                                    break
        elif asset_class == 'options':
            pass
        elif asset_class == 'equities':
            pass

    def sell_asset_class(self, asset_class, amount_to_trade):
        if asset_class == 'crypto':
            for symbol in self.crypto_symbols:
                current_price = self.get_current_price(symbol)
                if current_price:
                    qty = self.calculate_quantity(symbol)  # Calculate quantity using the appropriate method
                    if self.validate_trade(symbol, 'sell'):
                        try:
                            self.api.submit_order(
                                symbol=symbol,
                                qty=qty,
                                side='sell',
                                type='market',
                                time_in_force='gtc'
                            )
                            logging.info(f"Sold {qty} of {symbol} to rebalance crypto allocation.")
                            break
                        except tradeapi.rest.APIError as e:
                            if 'asset is not active' in str(e):
                                logging.warning(f"Skipping {symbol} because it is not active.")
                                continue
                            else:
                                logging.error(f"Failed to submit order for {symbol}: {e}")
                                raise e
        elif asset_class == 'options':
            pass
        elif asset_class == 'equities':
            pass

    def update_max_crypto_equity(self):
        # Get the current buying power of the account
        account = self.api.get_account()
        buying_power = float(account.buying_power)

        # Compute max_crypto_equity
        max_crypto_equity = buying_power

        # Update the JSON file with the new value
        self.risk_params['max_crypto_equity'] = max_crypto_equity

        return max_crypto_equity

    def check_options_risk(self, symbol, quantity, price):
        total_portfolio_value = float(self.api.get_account().portfolio_value)
        options_position_value = quantity * price
        options_allocation = options_position_value / total_portfolio_value

        if options_allocation > self.max_options_allocation:
            return False

        # Check individual options position loss
        if symbol in self.options_positions:
            entry_price = self.options_positions[symbol]['entry_price']
            if (price - entry_price) / entry_price < -self.max_options_loss_threshold:
                return False

        return True

    def monitor_options_expiration(self):
        positions = self.api.list_positions()
        current_date = datetime.now().date()

        for position in positions:
            if 'OPT' in position.symbol:
                expiration_date = position.expiration_date.date()
                days_to_expiration = (expiration_date - current_date).days

                if days_to_expiration <= 7:
                    # Close options position if it's within 7 days of expiration
                    self.close_position(position.symbol)
                    print(f"Closed options position {position.symbol} due to approaching expiration.")

    def monitor_volatility(self):
        # Retrieve market volatility data (e.g., VIX index)
        volatility = self.get_market_volatility()

        if volatility > self.risk_params['max_volatility_threshold']:
            # Adjust options positions based on high volatility
            self.adjust_options_positions(volatility)
            print(f"Adjusted options positions due to high market volatility: {volatility}")



    def calculate_position_greeks(self, position):
        # Extract the necessary parameters from the position object
        S = float(position.current_price)
        K = float(position.strike_price)
        T = (position.expiration_date - datetime.now()).days / 365  # Time to expiration in years
        r = 0.01  # Risk-free rate (adjust as needed)
        sigma = 0.20  # Volatility (adjust as needed)
        option_type = position.option_type.lower()

        # Calculate the Greeks
        greeks = calculate_greeks(S, K, T, r, sigma, option_type)
        return greeks

    def calculate_equity_allocation(self, asset_type='crypto'):
        risk_params = load_risk_params()
        if not risk_params:
            raise ValueError("Failed to load risk parameters.")

        max_crypto_equity = risk_params.get('max_crypto_equity', 0)
        max_equities_equity = risk_params.get('max_equity_equity', 0)

        equity = float(self.api.get_account().equity)
        positions = self.api.list_positions()

        total_crypto_value = sum(float(position.market_value) for position in positions if position.symbol.endswith('USD'))
        total_equities_value = sum(float(position.market_value) for position in positions if not position.symbol.endswith('USD'))
        total_options_value = sum(float(position.market_value) for position in positions if 'OPT' in position.symbol)

        print(f"Risk parameters loaded: {risk_params}")
        print(f"Total crypto value: {total_crypto_value}, Max crypto equity: {max_crypto_equity}")
        print(f"Total equities value: {total_equities_value}, Max equities equity: {max_equities_equity}")

        if asset_type == 'crypto':
            return max_crypto_equity
        elif asset_type == 'equity':
            return max_equities_equity
        elif asset_type == 'options':
            max_total_allocation = max_crypto_equity + max_equities_equity
            remaining_allocation = max_total_allocation - (total_crypto_value + total_equities_value + total_options_value)
            return remaining_allocation if remaining_allocation > 0 else 0
        else:
            raise ValueError("Invalid asset type specified. Choose either 'crypto', 'equity', or 'options'.")


    def optimize_portfolio(self, risk_aversion):
        # Get historical data for each symbol
        historical_data = {}
        for symbol in self.crypto_symbols:
            data, _ = alpha_vantage_crypto.get_digital_currency_daily(symbol=symbol, market='USD')
            historical_data[symbol] = data['4b. close (USD)']

        # Calculate expected returns and covariance matrix
        returns_data = pd.DataFrame(historical_data).pct_change()
        expected_returns = returns_data.mean()
        covariance_matrix = returns_data.cov()

        # Total investment amount
        total_investment = float(self.api.get_account().equity)

        # Run optimization in separate script
        quantities_to_purchase = optimize_portfolio(expected_returns, covariance_matrix, risk_aversion,
                                                    total_investment)

        return quantities_to_purchase

    def get_daily_returns(self, symbol: str, days: int = 3) -> float:
        url = "https://paper-api.alpaca.markets/v2/positions"
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
        }
        response = requests.get(url, headers=headers)
        data = response.json()

        # Find the position for the given symbol
        position_data = None
        for position in data:
            if position['symbol'] == symbol:
                position_data = position
                break

        if position_data is None:
            raise ValueError(f"No position found for symbol {symbol}")

        # Get the closing prices for the past `days` days
        closing_prices = []
        for _ in range(days):
            lastday_price = position_data.get('lastday_price')
            if lastday_price is not None:
                closing_prices.append(float(lastday_price))
            else:
                print(f"Missing lastday_price for symbol {symbol}. Skipping calculation of daily returns.")
                return []

        # Calculate the daily returns
        returns = [np.log(closing_prices[i] / closing_prices[i - 1]) for i in range(1, len(closing_prices))]

        # Return the daily returns
        return returns

    def get_position(self, symbol):
        """
        Get position details for a specific symbol
        """
        positions = self.api.list_positions()

        # Filter positions to find matches for the symbol
        symbol_positions = [p for p in positions if p.symbol == symbol]

        if not symbol_positions:
            print(f"No positions found for {symbol}")
            return None

        # Assuming there's only one position per symbol
        p = symbol_positions[0]

        # Get actual qty and unsettled qty
        actual_qty = float(p.qty)
        unsettled_qty = float(p.unsettled_qty) if hasattr(p,
                                                          'unsettled_qty') else 0  # Assuming 'unsettled_qty' is the correct attribute name

        pos = {
            "symbol": p.symbol,
            "qty": actual_qty,
            "unsettled_qty": unsettled_qty,
            "avg_entry_price": float(p.avg_entry_price) if p.avg_entry_price is not None else None
        }

        return pos

    total_trades_today = 0


    def calculate_position_values(self):
        positions = self.api.list_positions()
        self.crypto_value = 0.0
        self.commodity_value = 0.0
        # Calculate the total value of crypto and commodity positions
        for position in positions:
            symbol = position.symbol
            current_price = float(position.current_price)
            position_value = float(position.qty) * current_price

            if symbol.endswith('USD'):  # If the symbol ends with 'USD', it's a crypto position
                self.crypto_value += position_value
            else:  # Otherwise, it's a commodity position
                self.commodity_value += position_value

    def validate_trade(self, symbol, order_type):
        if self.total_trades_today >= 120:
            print("Hit daily trade limit, rejecting order")
            logging.info(f"Trade rejected for {symbol}: Hit daily trade limit")
            return False

        try:
            qty = self.calculate_quantity(symbol)

            if qty == 0:
                print(f"Calculated quantity for {symbol} is zero, rejecting order")
                logging.info(f"Trade rejected for {symbol}: Calculated quantity is zero")
                return False

            try:
                existing_position = self.get_position(symbol)
                current_qty = float(existing_position['qty']) if existing_position and existing_position[
                    'qty'] is not None else 0.0
            except Exception:
                current_qty = 0.0

            new_qty = float(current_qty) + float(qty)

            if new_qty > self.risk_params['max_position_size']:
                print(f'Original requested quantity for {symbol}: {new_qty}')
                print(f'Maximum position size according to risk parameters: {self.risk_params["max_position_size"]}')
                qty = self.risk_params['max_position_size'] - current_qty
                new_qty = self.risk_params['max_position_size']
                print(f'Adjusted quantity to comply with max position size: {new_qty}')
                logging.info(f"Adjusted buy quantity for {symbol} to {new_qty} to comply with max position size.")
            elif new_qty <= current_qty:
                print("No increase in position size, rejecting order")
                logging.info(
                    f"Trade rejected for {symbol}: No increase in position size (current_qty: {current_qty}, new_qty: {new_qty})")
                return False

            print(f"Running validation logic against trade for {symbol}...")

            portfolio = self.api.list_positions()

            asset_values = {
                'crypto': sum([float(p.current_price) * float(p.qty) for p in portfolio if
                               p.symbol.endswith('USD') and p.current_price is not None and p.qty is not None]),
                'commodity': sum([float(p.current_price) * float(p.qty) for p in portfolio if
                                  'commodity' in p.symbol and p.current_price is not None and p.qty is not None]),
                'equity': sum([float(p.current_price) * float(p.qty) for p in portfolio if
                               p.symbol.isalpha() and p.current_price is not None and p.qty is not None]),
                'options': sum([float(p.current_price) * float(p.qty) for p in portfolio if
                                'OPT' in p.symbol and p.current_price is not None and p.qty is not None])
            }

            portfolio_value = sum(asset_values.values())
            print(f"Current portfolio value (market value of all positions): ${round(portfolio_value, 2)}.")

            print('##################################################################')
            print('##################################################################')
            print('##################################################################')

            print('Retrieving the price details from the get_current_price method...')

            current_price = self.get_current_price(symbol)
            if current_price is None:
                print(
                    f"Failed to retrieve current price for {symbol} using primary method. Trying Alpaca API for options.")
                current_price = self.get_option_price_from_alpaca(symbol)

            if current_price is None:
                print(f"Failed to retrieve current price for {symbol} using Alpaca API.")
                logging.info(f"Trade rejected for {symbol}: Failed to retrieve current price.")
                return False

            print(f"Current Alpaca API price for {symbol} is: ${current_price}")

            proposed_trade_value = float(current_price) * float(qty)
            print(f"Total $ to purchase new order: ${round(proposed_trade_value, 2)}")

            open_orders = self.api.list_orders(status='open')
            open_symbols = [o.symbol for o in open_orders]

            account_cash = float(self.api.get_account().cash)
            print(f"Current account cash to buy: {account_cash}")

            print('##################################################################')
            print('##################################################################')
            print('##################################################################')

            print('Processing proposed_trade_value logic against current cash holdings...')

            if proposed_trade_value > account_cash:
                print("Proposed trade exceeds cash available to purchase crypto.")
                logging.info(
                    f"Trade rejected for {symbol}: Proposed trade value (${proposed_trade_value}) exceeds available cash (${account_cash}).")
                return False

            asset_class = None
            if 'USD' in symbol:
                asset_class = 'crypto'
            elif 'commodity' in symbol:
                asset_class = 'commodity'
            elif 'OPT' in symbol:
                asset_class = 'options'
            elif symbol.isalpha():
                asset_class = 'equity'

            if asset_class:
                if asset_class == 'crypto':
                    max_allocation = self.risk_params.get('max_crypto_equity', None)
                elif asset_class == 'equity':
                    max_allocation = self.risk_params.get('max_equity_equity', None)
                else:
                    max_allocation = self.calculate_equity_allocation(asset_type=asset_class)

                if max_allocation is not None:
                    print(f'Current {asset_class} equity: ${asset_values[asset_class]}')
                    print(f'Max {asset_class} equity: ${max_allocation}')

                    if float(asset_values[asset_class]) + float(proposed_trade_value) > max_allocation:
                        print(f"Trade exceeds max {asset_class} equity limit.")
                        logging.info(f"Trade rejected for {symbol}: Trade exceeds max {asset_class} equity limit.")
                        return False

            if order_type == 'buy':
                if qty > self.risk_params['max_position_size']:
                    print("Buy exceeds max position size")
                    logging.info(f"Trade rejected for {symbol}: Buy exceeds max position size.")
                    return False
            elif order_type == 'sell':
                position = self.get_position(symbol)
                position_qty = float(position['qty']) if position and position['qty'] is not None else 0.0
                qty = float(qty)
                if qty > position_qty:
                    print("Sell quantity exceeds position size")
                    logging.info(f"Trade rejected for {symbol}: Sell quantity exceeds position size.")
                    return False

            self.total_trades_today += 1
            return True
        except Exception as e:
            print(f"Error validating trade: {e}")
            logging.error(f"Error validating trade for {symbol}: {e}")
            return False

    def get_option_price_from_alpaca(self, symbol):
        url = f"https://paper-api.alpaca.markets/v2/options/contracts/{symbol}"
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
        }
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return float(data.get('close_price', 0))
            else:
                logging.error(f"Failed to get option price from Alpaca: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            logging.error(f"Error getting option price from Alpaca for {symbol}: {e}")
            return None

    def monitor_account_status(self):
        # Monitor and report on account status
        try:
            account = self.api.get_account()
            print(f"Equity: {account.equity}")
            print(f"Cash: {account.cash}")
            print(f"Buying Power: {account.buying_power}")
            return account
        except Exception as e:
            print(f"An exception occurred while monitoring account status: {str(e)}")
            return None

    def monitor_positions(self):
        # Monitor and report on open positions
        try:
            positions = self.api.list_positions()
            for position in positions:
                pos_details = self.get_position(position.symbol)
                if pos_details:
                    print(
                        f"Symbol: {pos_details['symbol']}, Quantity: {pos_details['qty']}, Avg Entry Price: {pos_details['avg_entry_price']}")
            return positions
        except Exception as e:
            print(f"An exception occurred while monitoring positions: {str(e)}")
            return None

    def get_crypto_fee(self, volume):
        if volume < 100_000:
            return 0.0025
        elif volume < 500_000:
            return 0.0022
        elif volume < 1_000_000:
            return 0.002
        elif volume < 10_000_000:
            return 0.0018
        elif volume < 25_000_000:
            return 0.0015
        elif volume < 50_000_000:
            return 0.0013
        elif volume < 100_000_000:
            return 0.0012
        else:
            return 0.001

    def report_profit_and_loss(self):
        url = "https://paper-api.alpaca.markets/v2/account"
        url_portfolio_history = "https://paper-api.alpaca.markets/v2/account/portfolio/history"
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
        }

        try:
            # Get account data
            account_response = requests.get(url, headers=headers)
            account_data = account_response.json()
            cash_not_invested = float(account_data['cash'])

            # Get portfolio history data
            portfolio_history_response = requests.get(url_portfolio_history, headers=headers)
            portfolio_history_data = portfolio_history_response.json()

            # Filter out 'None' values
            equity_values = [v for v in portfolio_history_data['equity'] if v is not None]

            # Calculate PnL based on portfolio history
            first_equity = float(equity_values[0])  # First equity value
            last_equity = float(equity_values[-1])  # Last equity value
            commissions = first_equity * 0.01

            print(f'First equity is: {first_equity}.')
            print(f'Last equity is: {last_equity}.')
            print(f'Total commisions were: {commissions}.')

            # find pnl for account
            pnl_total = last_equity - first_equity - commissions

            # find total equity for reporting
            total_equity = pnl_total + cash_not_invested

            print(
                f"Total Profit/Loss: ${round(pnl_total,2)}. Total equity (cash invested plus cash not invested): ${round(total_equity,2)}")
            return pnl_total

        except Exception as e:
            print(f"An exception occurred while reporting profit and loss: {str(e)}")
            return 0

    def get_equity(self):
        return float(self.api.get_account().equity)

    def update_risk_parameters(self):
        # Dynamically adjust risk parameters based on account performance
        pnl_total = self.report_profit_and_loss()
        account = self.api.get_account()
        current_equity = float(account.equity)

        self.risk_params['max_portfolio_size'] = current_equity  # Update the max_portfolio_size with the current equity

        if pnl_total is None:
            print("Could not calculate PnL, not updating risk parameters.")
            return

        pnl_total = float(round(pnl_total, 2))
        print(f'pnl is accurately: {pnl_total}')

        if pnl_total <= 0:
            print("PnL is negative...")
            if self.risk_params['max_position_size'] >= 50:
                print("Reducing risk parameters...")
                self.risk_params['max_position_size'] *= 0.90  # reduce by 10%
                self.risk_params['max_portfolio_size'] *= 0.90  # reduce by 10%
            else:
                print("Max position size is less than 50. Not reducing risk parameters.")
        elif pnl_total > 0:
            print("PnL is positive...")
            if self.risk_params['max_position_size'] >= 50:
                print("Increasing risk parameters...")
                self.risk_params['max_position_size'] *= 1.0015  # increase by .15%
                self.risk_params['max_portfolio_size'] *= 1.0015  # increase by .15%
            else:
                print("Max position size is less than 50. Not increasing risk parameters.")
        else:
            print("PnL is neutral, no changes to risk parameters.")

        with open('risk_params.json', 'w') as f:
            json.dump(self.risk_params, f)
        print("Risk parameters updated.")
        return self.risk_params

    def calculate_drawdown(self):
        try:
            portfolio = self.api.list_positions()
            portfolio_value = sum([float(position.current_price) * float(position.qty) for position in portfolio])

            # Update peak portfolio value if current portfolio value is higher
            if portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = portfolio_value

            # Calculate drawdown if portfolio is not empty
            if portfolio_value > 0 and self.peak_portfolio_value > 0:
                drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            else:
                drawdown = 0

            return drawdown
        except Exception as e:
            print(f"An exception occurred while calculating drawdown: {str(e)}")
            return None

    def check_risk_before_order(self, symbol, new_shares):
        """
        Check the risk parameters before placing an order.

        The function will prevent an order if the new shares would result in a position size
        that violates the risk parameters.
        """
        # Get the current position
        try:
            current_position = self.api.get_position(symbol)
            current_shares = float(current_position.qty)
        except:
            current_shares = 0

        # Calculate the new quantity of shares after the purchase
        total_shares = current_shares + float(new_shares)

        # Check if the new quantity violates the risk parameters
        if total_shares > self.risk_params['max_position_size']:
            return 'Order exceeded permissible balance. Total order share exceed max position size allowable.'
            # If the new quantity violates the max position size, prevent the order
            return False
        else:
            # If the new quantity doesn't violate the risk parameters, adjust the quantity and place the order
            delta_shares = self.risk_params['max_position_size'] - current_shares

            if delta_shares > 0:
                # Get the average entry price
                avg_entry_price = self.get_avg_entry_price(symbol)

                if avg_entry_price is not None and avg_entry_price != 0:
                    # Calculate the adjusted quantity based on the average entry price
                    adjusted_quantity = int(delta_shares / avg_entry_price)

                    # Place the order with the adjusted quantity
                    self.api.submit_order(
                        symbol=symbol,
                        qty=adjusted_quantity,
                        side='buy',
                        type='limit',
                        time_in_force='gtc',
                        limit_price=avg_entry_price
                    )

            return True

    def check_momentum(self, symbol, momentum_signal):
        """
        Checks the momentum signal and decides whether to sell the entire position.
        """
        # Get position
        position_list = [position for position in self.api.list_positions() if position.symbol == symbol]

        if len(position_list) == 0:
            print(f"No position exists for {symbol}.")
            return

        position = position_list[0]

        # If momentum signal is 'Sell' and the percentage change is negative, sell the entire position
        if momentum_signal == "Sell" and float(position.unrealized_plpc) < 0:
            qty = position.qty
            if self.validate_trade(symbol, qty, "sell"):
                # Place a market sell order
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"Selling the entire position of {symbol} due to negative momentum.")

    def calculate_quantity(self, symbol):
        """
        Calculates the quantity to purchase based on available equity and current price.
        """
        # Get account info
        account = self.api.get_account()
        available_cash = float(account.cash)

        risk_params = load_risk_params()

        max_crypto_equity = float(risk_params['max_crypto_equity'])

        # Calculate the current total investment in cryptocurrencies
        positions = self.api.list_positions()
        total_investment = sum([float(p.market_value) for p in positions if p.symbol.endswith('USD')])

        # If total investment is already at or exceeds max_crypto_equity, return quantity 0
        if total_investment >= max_crypto_equity:
            print(
                f"Total investment in cryptocurrencies is already at or exceeds the maximum permitted. Returning quantity 0.")
            return 0

        # Calculate allowable investment as max_crypto_equity minus total investment
        allowable_investment = max_crypto_equity - total_investment

        # Determine investable amount
        investable_amount = min(available_cash, allowable_investment)

        # Check if investable amount is less than 1
        if investable_amount < 1:
            print(f"Investable amount for {symbol} is less than 1. Returning quantity 0.")
            return 0

        # Use the current price
        current_price = self.get_current_price(symbol)
        if current_price == 0 or current_price is None:
            return 0

        # Calculate a preliminary quantity based on the available cash
        preliminary_quantity = investable_amount / current_price

        # Tiered system for quantity adjustment
        if current_price > 4001:  # High-priced assets like BTC
            quantity = preliminary_quantity * 0.01  # buy less of high-priced assets
        elif 3001 < current_price <= 4000:  # Mid-priced assets
            quantity = preliminary_quantity * 0.0354
        elif 1000 < current_price <= 3000:  # Mid-priced assets
            quantity = preliminary_quantity * 0.0334
        elif 201 < current_price <= 999:  # Mid-priced assets
            quantity = preliminary_quantity * 0.04534
        elif 20 < current_price <= 200:  # Mid-priced assets
            quantity = preliminary_quantity * 0.09434
        elif -0.000714 < current_price <= 20.00:  # Mid-priced assets
            quantity = preliminary_quantity * 0.031434
        else:  # ordinary priced assets
            quantity = preliminary_quantity  # buy more of low priced assets

        quantity = round(quantity, 5)

        print(f"Calculated quantity for {symbol}: {quantity}")
        return quantity

    def execute_profit_taking(self, symbol, pct_gain=0.05):
        """
        Executes a profit-taking strategy.
        If the profit for a specific crypto reaches a certain percentage, sell enough shares to realize the profit.
        """
        position_list = [position for position in self.api.list_positions() if position.symbol == symbol]

        if len(position_list) == 0:
            print(f"No position exists for {symbol}.")
            return

        position = position_list[0]

        # If the unrealized profit percentage is greater than the specified percentage, sell a portion of the position
        if float(position.unrealized_plpc) > pct_gain:
            qty = int(float(position.qty) * pct_gain)  # Selling enough shares to realize the 5% gain

            if self.validate_trade(symbol, qty, "sell"):
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"Selling {qty} shares of {symbol} to realize profit.")

    def execute_stop_loss(self, symbol, pct_loss=0.07):
        """
        Executes a stop-loss strategy.
        If the loss for a specific crypto reaches a certain percentage, sell the entire position.
        """
        position_list = [position for position in self.api.list_positions() if position.symbol == symbol]

        if len(position_list) == 0:
            logging.info(f"No position exists for {symbol}.")
            return

        position = position_list[0]

        # If the unrealized loss percentage is greater than the specified percentage, sell the entire position
        unrealized_loss_pct = float(position.unrealized_plpc)
        logging.info(
            f"Checking stop-loss for {symbol}: Unrealized Loss {unrealized_loss_pct}% vs. Threshold {pct_loss * 100}%")
        if unrealized_loss_pct < -pct_loss:
            logging.info(f"Unrealized loss for {symbol} exceeds {pct_loss}%: {unrealized_loss_pct}%")
            qty = position.qty

            if self.validate_trade(symbol, qty, "sell"):
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                logging.info(f"Selling the entire position of {symbol} due to stop loss.")
            else:
                logging.info(f"Trade validation failed for {symbol} with quantity {qty}.")
        else:
            logging.info(f"Stop-loss condition not met for {symbol}. Current loss: {unrealized_loss_pct}%")

    # Define the calculate_profitability function
    @staticmethod
    def calculate_profitability(current_price, avg_entry_price):
        if avg_entry_price == 0:
            logging.error("Average entry price is zero. Cannot calculate profitability.")
            return None
        return ((current_price - avg_entry_price) / avg_entry_price) * 100


    def enforce_diversification(self, symbol, max_pct_portfolio=0.30):
        """
        Enforces diversification by ensuring that no crypto makes up more than a certain percentage of the portfolio.
        """
        portfolio = self.api.list_positions()
        portfolio_value = sum([float(position.current_price) * float(position.qty) for position in portfolio])
        position_list = [position for position in portfolio if position.symbol == symbol]

        if len(position_list) == 0:
            print(f"No position exists for {symbol}.")
            return

        position = position_list[0]
        position_value = float(position.current_price) * float(position.qty)

        # If the value of this position exceeds the maximum percentage of the portfolio, sell enough shares to get below the maximum
        if position_value / portfolio_value > max_pct_portfolio:
            excess_value = position_value - (portfolio_value * max_pct_portfolio)
            qty_to_sell = int(excess_value / float(position.current_price))

            if self.validate_trade(symbol, qty_to_sell, "sell"):
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty_to_sell,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"Selling {qty_to_sell} shares of {symbol} to maintain diversification.")

    def generate_momentum_signal(self, symbol):
        """
        Generate a momentum signal for the given symbol.

        Returns "Buy" if the symbol has increased in value by 5% or more since purchase,
        and "Sell" if it has decreased by 7% or more. Otherwise, returns "Hold".
        """
        # Get the purchase price for this stock
        # TODO: Replace this with your own logic
        purchase_price = self.get_purchase_price(symbol)

        # Get the current price for this stock
        current_price = self.get_avg_entry_price(symbol)

        # Calculate the percentage change since purchase
        pct_change = (current_price - purchase_price) / purchase_price * 100

        # Generate the momentum signal
        if pct_change >= 5:
            return "Buy"
        elif pct_change <= -7:
            return "Sell"
        else:
            return "Hold"



    def get_purchase_price(self, symbol):
        """
        Retrieve the purchase price of the given symbol.
        """
        trades = download_trades()

        # Filter trades for the given symbol
        trades = [trade for trade in trades if trade[0] == symbol]

        if not trades:
            return None

        # Get the last trade for the symbol
        last_trade = trades[-1]

        # The price is the third element in the trade
        return float(last_trade[2])

    def get_avg_entry_price(self, symbol):
        try:
            position = self.api.get_position(symbol)
            avg_entry_price = float(position.avg_entry_price)
            print(f"For symbol {symbol}, average entry price is {avg_entry_price}.")
            return avg_entry_price
        except Exception as e:
            print(f"No position in {symbol} to calculate average entry price. Error: {str(e)}")
            return 0

    ## this section provides logic to pull the latest price of something and use that
    @staticmethod
    def fetch_exchange_rate(base_currency, quote_currency):
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={base_currency}&to_currency={quote_currency}&apikey={ALPHA_VANTAGE_API}"
        try:
            response = requests.get(url).json()
            exchange_rate = response["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
            return float(exchange_rate)
        except KeyError:
            print(f"No exchange rate found from {base_currency} to {quote_currency}")
            return None
        except Exception as e:
            print(f"Error while fetching exchange rate: {e}")
            return None

    @staticmethod
    def fetch_and_process_data(base_crypto, quote='USD'):
        exchange_rate = RiskManagement.fetch_exchange_rate(base_crypto, quote)

        if exchange_rate is None:
            print(f"No exchange rate found for {base_crypto} to {quote}")
            return None

        url = f"https://www.alphavantage.co/query?function=CRYPTO_INTRADAY&symbol={base_crypto}&market={quote}&interval=5min&outputsize=compact&apikey={ALPHA_VANTAGE_API}"
        try:
            response = requests.get(url).json()
            intraday_data = response.get('Time Series Crypto (5min)', {})
            if not intraday_data:
                print(f"No intraday data found for {base_crypto}/{quote}")
                return None

            latest_intraday_data = list(intraday_data.items())[0]  # Get the latest 5-minute interval
            latest_price = float(latest_intraday_data[1]['4. close']) * exchange_rate
            return latest_price
        except Exception as e:
            print(f"Error while fetching intraday stats: {e}")
            return None

    def get_current_price(self, symbol):
        print(f'Running current price function lookup: {symbol}')

        # Check if symbol includes "USD"
        if symbol.endswith("USD"):
            if '/' in symbol:
                base_crypto, quote_currency = symbol.split('/')  # Split symbol to get base cryptocurrency
            else:
                base_crypto = symbol[:-3]  # Extract base cryptocurrency from symbol
                quote_currency = "USD"

            if quote_currency != "USD":
                print(f"Symbol {symbol} is not in the correct format with USD as the quote currency.")
                return 0

            try:
                print(f"Fetching price for {symbol} using Alpha Vantage API.")
                current_price = self.fetch_and_process_data(base_crypto, 'USD')
                if current_price:
                    print(f"Current price for {symbol} is {current_price}.")
                    return current_price
                else:
                    print(f"Failed to fetch current price for {symbol} using Alpha Vantage.")
                    return 0
            except Exception as e:
                print(f"Failed to get current price for {symbol} from Alpha Vantage. Error: {str(e)}")
                return 0

        # If not ending with "USD", attempt to fetch price from Alpha Vantage (stocks)
        try:
            print('Trying Alpha Vantage API for stock symbols')
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API}"
            response = requests.get(url)
            data = response.json()

            last_update = list(data['Time Series (Daily)'].keys())[0]
            current_price_str = data['Time Series (Daily)'][last_update]['4. close']
            current_price = float(current_price_str)

            print(
                f"Current price for {symbol} is {current_price} (string value: {current_price_str}) at {last_update}.")
            return current_price
        except Exception as e:
            print(f'Cannot find {symbol} in Alpha Vantage: {e}')
            return 0


    def analyze_trend(self, symbol):
        try:
            # Fetch historical data
            data, _ = alpha_vantage_ts.get_daily(symbol=symbol, outputsize='compact')

            # Calculate moving averages
            data['SMA_20'] = data['4. close'].rolling(window=20).mean()
            data['SMA_50'] = data['4. close'].rolling(window=50).mean()

            # Determine trend
            if data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1]:
                return 'uptrend'
            elif data['SMA_20'].iloc[-1] < data['SMA_50'].iloc[-1]:
                return 'downtrend'
            else:
                return 'neutral'
        except Exception as e:
            logging.error(f"Error analyzing trend for {symbol}: {e}")
            return 'neutral'

    def monitor_and_close_expiring_options(self):
        try:
            positions = self.api.list_positions()
            if not positions:
                logging.info("No positions to monitor.")
                return

            current_date = datetime.now().date()

            for position in positions:
                if 'OPT' in position.symbol:
                    expiration_date = datetime.strptime(position.expiration_date, '%Y-%m-%d').date()
                    days_to_expiration = (expiration_date - current_date).days

                    if days_to_expiration <= 7:
                        # Analyze the trend
                        trend = self.analyze_trend(
                            position.symbol.replace('OPT', ''))  # Adjust symbol for trend analysis

                        # Decision based on trend
                        if trend == 'downtrend':
                            # Close options position
                            qty = position.qty
                            self.api.submit_order(
                                symbol=position.symbol,
                                qty=qty,
                                side='sell',
                                type='market',
                                time_in_force='gtc'
                            )
                            logging.info(
                                f"Closed options position {position.symbol} due to approaching expiration and downtrend.")
                            print(
                                f"Closed options position {position.symbol} due to approaching expiration and downtrend.")
                        else:
                            logging.info(f"Decided not to sell {position.symbol} due to trend: {trend}")
                            print(f"Decided not to sell {position.symbol} due to trend: {trend}")
        except Exception as e:
            logging.error(f"Error in monitor_and_close_expiring_options: {e}")
            print(f"Error in monitor_and_close_expiring_options: {e}")



    TARGET_ALLOCATION = {
        'options': 0.20,  # 20%
        'crypto': 0.30,  # 30%
        'equities': 0.50  # 50%
    }


def get_alpha_vantage_data(base_currency, quote_currency):
    url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={base_currency}&to_currency={quote_currency}&apikey={ALPHA_VANTAGE_API}"

    response = requests.get(url)
    data = response.json()

    if "Realtime Currency Exchange Rate" in data:
        # Get the exchange rate
        exchange_rate = data["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
        return exchange_rate
    else:
        print("Error getting data from Alpha Vantage")
        return None

def send_teams_message(teams_url, message):
    headers = {
        "Content-type": "application/json",
    }
    response = requests.post(teams_url, headers=headers, data=json.dumps(message))
    return response.status_code


def get_profit_loss(positions):
    profit_loss = 0
    for position in positions:
        profit_loss += (position['current_price'] - position['avg_entry_price']) * float(position['quantity'])
    return profit_loss



facts = []

if __name__ == "__main__":
    risk_manager = RiskManagement(api, risk_params)

    current_allocation = risk_manager.calculate_current_allocation()
    risk_manager.rebalance_portfolio()

    # Monitor expiring options and close them based on trend analysis
    risk_manager.monitor_and_close_expiring_options()

    # Other existing logic
    risk_manager.monitor_account_status()
    risk_manager.monitor_positions()
    risk_manager.report_profit_and_loss()
    risk_manager.update_risk_parameters()

    # Fetch account and portfolio information
    account = risk_manager.api.get_account()
    portfolio = risk_manager.api.list_positions()

    # Calculate and display allowed equity for crypto and commodities
    commodity_equity = risk_manager.calculate_equity_allocation(asset_type='equity')
    crypto_equity = risk_manager.calculate_equity_allocation(asset_type='crypto')
    print(f"Total Allowed Commodity Equity: {commodity_equity}")
    print(f"Total Allowed Crypto Equity: {crypto_equity}")

    # Initialize portfolio summary
    portfolio_summary = {
        'equity': float(account.equity),
        'cash': float(account.cash),
        'buying_power': float(account.buying_power),
        'positions': []
    }

    # Iterate over each position to calculate details
    for position in portfolio:
        symbol = position.symbol
        avg_entry_price = float(position.avg_entry_price)
        current_price = risk_manager.get_current_price(symbol)  # Assumes get_current_price is correctly implemented
        qty = float(position.qty)

        try:
            profitability = RiskManagement.calculate_profitability(current_price, avg_entry_price)
            if profitability is not None:
                portfolio_summary['positions'].append({
                    'symbol': symbol,
                    'avg_entry_price': avg_entry_price,
                    'current_price': current_price,
                    'profitability': float(profitability),
                    'quantity': qty
                })
            else:
                portfolio_summary['positions'].append({
                    'symbol': symbol,
                    'avg_entry_price': avg_entry_price,
                    'current_price': current_price,
                    'profitability': None,
                    'quantity': qty
                })
        except ZeroDivisionError:
            logging.error(f"Division by zero error for symbol {symbol} with avg_entry_price {avg_entry_price}.")
            portfolio_summary['positions'].append({
                'symbol': symbol,
                'avg_entry_price': avg_entry_price,
                'current_price': current_price,
                'profitability': None,
                'quantity': qty
            })

    # Calculate profit/loss for the portfolio
    portfolio_summary['profit_loss'] = sum(
        (pos['current_price'] - pos['avg_entry_price']) * pos['quantity']
        for pos in portfolio_summary['positions']
        if pos['current_price'] is not None and pos['avg_entry_price'] is not None
    )

    # Update risk parameters based on profit/loss
    portfolio_summary['risk_parameters_updated'] = portfolio_summary['profit_loss'] > 0