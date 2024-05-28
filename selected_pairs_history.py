import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import time
from datetime import datetime
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import logging
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# Additional Filters and Criteria

# Profit Margin:
# - Very low or Negative: A profit margin of 1% or lower is considered very low,
#   and if it's negative (e.g., -5%), the company is losing money.
#   Example: Amazon, in its early years, often reported negative profit margins.
# - High: A profit margin above 15% is typically considered high.
#   Example: Companies like Microsoft often have profit margins exceeding 30%.

# Price-Earnings Ratio (P/E):
# - Very low or Negative: A P/E ratio below 5 is considered low, suggesting
#   the market has low expectations for the company's future. Companies with negative earnings have a negative P/E ratio.
#   Example: In 2020, many airlines had negative P/E ratios due to substantial losses caused by the COVID-19 pandemic.
# - High: A P/E ratio above 20 is typically considered high, indicating that
#   the market expects high earnings growth.
#   Example: Amazon has had a high P/E ratio for many years, often exceeding 100.

# Return on Equity (ROE):
# - Very low or Negative: An ROE below 5% is considered low, suggesting the company
#   isn't generating much profit from its equity. Negative ROE (e.g., -10%) means the company is losing money.
#   Example: In 2008 during the financial crisis, many banks reported negative ROE.
# - High: An ROE above 20% is generally considered high.
#   Example: Companies like Apple have consistently reported ROE above 30%.

# EV to EBITDA:
# - Very low or Negative: An EV/EBITDA below 5 is generally considered low, suggesting
#   the company might be undervalued, assuming it's a profitable business. Negative values can occur if EBITDA is negative,
#   indicating operating losses. Example: In 2008, during the financial crisis, some banks had low EV/EBITDA ratios.
# - High: An EV/EBITDA above 15 is usually considered high, suggesting the company may be overvalued.
#   High-growth tech companies often have high EV/EBITDA ratios. Example: Zoom Video Communications had an EV/EBITDA ratio over 200 in 2020.

# Quarterly Earnings Growth YoY:
# - Very low or Negative: Negative quarterly earnings growth means the company's earnings have shrunk compared to the same quarter in the previous year.
#   Example: During the COVID-19 pandemic in 2020, many companies in the travel and hospitality industry faced negative quarterly earnings growth.
# - High: A high number (e.g., 50% or higher) would indicate a significant increase in earnings compared to the same quarter in the previous year.
#   Example: Many tech companies like Apple and Amazon reported high quarterly earnings growth in 2020 due to the increased demand for digital services amidst the pandemic.

# Load environment variables and set up logging
# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database configuration
MONGO_DB_CONN_STRING = os.getenv("MONGO_DB_CONN_STRING")
DB_NAME = "stock_data"

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API")
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
ti = TechIndicators(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

# MongoDB connection
client = MongoClient(MONGO_DB_CONN_STRING)
db = client[DB_NAME]
company_overviews_collection = db["company_overviews"]
selected_pairs_collection = db["selected_pairs"]

def get_current_price_and_sma(symbol, period=20):
    try:
        daily_data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
        sma_data, _ = ti.get_sma(symbol=symbol, interval='daily', time_period=period)
        current_price = daily_data['4. close'].iloc[-1]
        sma_value = sma_data['SMA'].iloc[-1]
        return symbol, current_price, sma_value
    except Exception as e:
        logging.error(f"Error fetching price and SMA for {symbol}: {e}")
        return symbol, np.nan, np.nan

def get_rsi(symbol, period=14):
    try:
        rsi_data, _ = ti.get_rsi(symbol=symbol, interval='daily', time_period=period)
        rsi_value = rsi_data['RSI'].iloc[-1]
        return symbol, rsi_value
    except Exception as e:
        logging.error(f"Error fetching RSI for {symbol}: {e}")
        return symbol, np.nan

def fetch_company_overviews():
    try:
        logging.info("Fetching company overviews...")
        documents = company_overviews_collection.find({}, {'_id': 0})
        df = pd.DataFrame(list(documents))
        if df.empty:
            logging.info("Fetched DataFrame is empty.")
        else:
            logging.info(f"Fetched {len(df)} documents.")
            # Convert specified columns to numeric, coerce errors to NaN
            numeric_cols = ["ProfitMargin", "PERatio", "ReturnOnEquityTTM", "EVToEBITDA", "QuarterlyEarningsGrowthYOY"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        logging.error(f"An error occurred while fetching company overviews: {e}")
        return pd.DataFrame()

def test_db_connection():
    try:
        count = company_overviews_collection.count_documents({})
        logging.info(f"Test query found {count} documents in 'company_overviews'.")
    except Exception as e:
        logging.error(f"Test query failed: {e}")


from datetime import datetime


def store_selected_pairs(df):
    # Adding a 'date_added' field to each document
    df['date_added'] = datetime.now().strftime('%Y-%m-%d')

    for _, row in df.iterrows():
        # Creating a filter for an existing document with the same symbol and date_added
        filter_ = {'symbol': row['Symbol'], 'date_added': row['date_added']}
        # Converting the row to a dictionary, excluding the index
        update = {'$set': row.to_dict()}

        # Performing an upsert operation: update if exists, insert if not
        result = selected_pairs_collection.update_one(filter_, update, upsert=True)

        if result.upserted_id:
            logging.info(f"Inserted new document for symbol: {row['Symbol']}")
        elif result.modified_count > 0:
            logging.info(f"Updated document for symbol: {row['Symbol']}")
        else:
            logging.info(f"No changes for symbol: {row['Symbol']}")

    logging.info("Selected pairs updated in MongoDB 'selected_pairs' collection.")


def analyze_data_before_filtering(df):
    # Descriptive statistics before applying filters
    logging.info("Descriptive statistics before filtering:")
    logging.info(df[["ProfitMargin", "PERatio", "ReturnOnEquityTTM", "EVToEBITDA", "QuarterlyEarningsGrowthYOY"]].describe())

    # Histograms for visualizing distributions
    for col in ["ProfitMargin", "PERatio", "ReturnOnEquityTTM", "EVToEBITDA", "QuarterlyEarningsGrowthYOY"]:
        plt.figure(figsize=(10, 6))
        df[col].hist(bins=20)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(False)
        plt.show()

def fetch_financial_data(symbol):
    current_price, sma_value = get_current_price_and_sma(symbol)
    _, rsi_value = get_rsi(symbol)
    return symbol, current_price, sma_value, rsi_value

def main():
    test_db_connection()
    logging.info("Starting script...")
    df = fetch_company_overviews()
    if df.empty:
        logging.info("Exiting script due to no data.")
        return

    # Analyze data before applying filters
    #analyze_data_before_filtering(df)

    # Apply updated filters based on financial metrics and growth focus
    df_filtered = df[
        (df["ProfitMargin"] > -10) &  # Looser constraint to allow growth-focused companies
        (df["PERatio"] > 0) &  # Ensure positive earnings, but no upper limit to allow high-growth companies
        (df["ReturnOnEquityTTM"] > -20) &  # Broader range to include companies investing heavily in growth
        (df["EVToEBITDA"].between(0, 50)) &  # Allow higher ratios typical of growth companies
        (df["QuarterlyEarningsGrowthYOY"] > 0)  # Focus on companies with positive earnings growth
        ]

    # Analyze data after applying filters
    logging.info("Descriptive statistics after filtering:")
    logging.info(df_filtered[["ProfitMargin", "PERatio", "ReturnOnEquityTTM", "EVToEBITDA", "QuarterlyEarningsGrowthYOY"]].describe())

    # Sort by Market Capitalization and select pairs
    selected_pairs = df_filtered.sort_values("MarketCapitalization", ascending=False).head(20)

    # Store the selected pairs in MongoDB
    store_selected_pairs(selected_pairs)

    # Use multi-threading to fetch financial data
    symbols = selected_pairs['Symbol'].tolist()
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_symbol = {executor.submit(fetch_financial_data, symbol): symbol for symbol in symbols}


if __name__ == "__main__":
    main()