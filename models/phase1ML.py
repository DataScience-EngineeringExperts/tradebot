import pandas as pd
from pymongo import MongoClient, UpdateOne
from datetime import datetime
from dotenv import load_dotenv
import os


# Load environment variables and establish MongoDB connections
load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

client = MongoClient(MONGO_DB_CONN_STRING)
db_stock_data = client['stock_data']
db_silvertables = client['silvertables']
db_ml_ready = client['machinelearning']

# MongoDB collections
aggregated_stock_data = db_stock_data['aggregated_stock_data']
ML_sentiment_data = db_silvertables['ML_sentiment_data']

# Create index on 'symbol' field
aggregated_stock_data.create_index('symbol')

def fetch_time_series_data(symbol, aggregated_collection):
    print(f"Fetching data for symbol: {symbol}")
    query = {'symbol': symbol}
    cursor = aggregated_collection.find(query, {'_id': 0, 'symbol': 1, 'TIME_SERIES_DAILY_data': 1})
    data = list(cursor)
    if data and 'TIME_SERIES_DAILY_data' in data[0]:
        time_series_data = data[0]['TIME_SERIES_DAILY_data']
        df = pd.DataFrame(time_series_data)
        df.rename(columns={'timestamp': 'date', 'close_price': 'close'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        print(f"Data fetched and processed for {symbol}.")
        return df
    else:
        print(f"No 'TIME_SERIES_DAILY_data' found for symbol: {symbol}.")
        return None

def integrate_sentiment_data(df_time_series, symbol, db_silvertables):
    print(f"Integrating sentiment data for symbol: {symbol}")
    collection = db_silvertables['ML_sentiment_data']
    doc = collection.find_one({"symbol": symbol})

    if doc and 'cleaned_data' in doc:
        sentiment_data = pd.DataFrame([doc['cleaned_data']])
        sentiment_date = pd.to_datetime(doc['datetime_processed'])
        sentiment_data['date'] = sentiment_date
        sentiment_data.set_index('date', inplace=True)
        sentiment_data = sentiment_data.reindex(df_time_series.index, method='ffill')

        # Add collection prefix to avoid duplicate columns
        sentiment_data.columns = [f"sentiment_{col}" for col in sentiment_data.columns]

        df_time_series = pd.merge(df_time_series, sentiment_data, left_index=True, right_index=True, how='left')
        print(f"Sentiment data integrated for {symbol}.")
    else:
        print(f"No sentiment data found for symbol: {symbol}")

    return df_time_series


def store_data(df, symbol, db_ml_ready):
    collection_name = f"{symbol}_processed_data"
    collection = db_ml_ready[collection_name]

    # Replace NaN values with 0 and ensure proper data types
    df = df.fillna(0).infer_objects()  # Infer objects tries to convert object types to more specific types like float or int

    df.reset_index(inplace=True)  # Reset index to turn the 'date' from index to a column
    data_dict = df.to_dict("records")

    # Metadata for traceability
    current_time = datetime.now()
    for record in data_dict:
        record['symbol'] = symbol
        record['date_imported'] = current_time

    # Bulk write operation for upsert
    bulk_operations = [
        UpdateOne(
            {'symbol': symbol, 'date': record['date']},
            {'$set': record},
            upsert=True
        )
        for record in data_dict
    ]
    collection.bulk_write(bulk_operations)
    print(f"Data for {symbol} stored in MongoDB collection '{collection_name}'.")



def main():
    symbols = aggregated_stock_data.distinct('symbol')
    for symbol in symbols:
        print(f"Processing {symbol}...")
        df_time_series = fetch_time_series_data(symbol, aggregated_stock_data)
        if df_time_series is not None:
            df_time_series = integrate_sentiment_data(df_time_series, symbol, db_silvertables)
            store_data(df_time_series, symbol, db_ml_ready)
        else:
            print(f"Skipping {symbol} due to missing data.")
        print(f"Completed processing for {symbol}")

if __name__ == "__main__":
    main()