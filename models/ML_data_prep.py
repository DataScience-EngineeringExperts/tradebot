import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import datetime
from datetime import datetime, timezone


class ETLPipeline:
    def __init__(self, mongo_conn_string):
        self.client = MongoClient(mongo_conn_string)
        self.db_source = self.client.stock_data
        self.db_target = self.client.silvertables

    def clean_and_transform(self, doc, source_collection_name):
        if source_collection_name in ['balance_sheet', 'cash_flow', 'technicals']:
            if source_collection_name == 'balance_sheet':
                data_items = doc['balance_sheet']['annualReports']
            elif source_collection_name == 'cash_flow':
                data_items = doc['cash_flow']['annualReports']
            else:
                data_items = doc['income_statement']['annualReports']

            df = pd.DataFrame(data_items)
            numeric_cols = df.columns.drop(['fiscalDateEnding', 'reportedCurrency'])
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

            required_cols = ['fiscalDateEnding', 'reportedCurrency']
            if not set(required_cols).issubset(df.columns):
                raise ValueError(f"Missing required columns in data for symbol: {doc['symbol']}")

            return df.to_dict('records')
        elif source_collection_name == 'news_sentiment_data':
            cleaned_data = {
                'overall_sentiment': doc['overall_sentiment'],
                'ticker_sentiment': doc['ticker_sentiment'],
                'sentiment_by_source': doc['sentiment_by_source'],
                'sentiment_by_topic': doc['sentiment_by_topic']
            }
            return cleaned_data
        else:
            raise ValueError(f"Unsupported source collection: {source_collection_name}")

    def fetch_clean_store(self, source_collection_name, target_collection_name):
        source_collection = self.db_source[source_collection_name]
        target_collection = self.db_target[target_collection_name]

        doc_count = source_collection.count_documents({})
        print(f"Fetched {doc_count} documents from source collection: {source_collection_name}")

        all_docs = source_collection.find({})
        processed_docs = []

        for doc in all_docs:
            try:
                cleaned_data = self.clean_and_transform(doc, source_collection_name)
                new_doc = {
                    "symbol": doc['symbol'],
                    "cleaned_data": cleaned_data,
                    "datetime_processed": datetime.now(timezone.utc),
                    "version": 1
                }
                processed_docs.append(new_doc)
                print(f"Processed data for symbol: {doc['symbol']}")
            except ValueError as ve:
                print(f"Error processing data for symbol: {doc['symbol']}")
                print(f"Error message: {str(ve)}")

        if processed_docs:
            target_collection.insert_many(processed_docs)
            print(f"Stored {len(processed_docs)} documents in the target collection: {target_collection_name}")
        else:
            print(f"No valid documents to store in the target collection: {target_collection_name}")


# Usage
if __name__ == '__main__':
    load_dotenv()
    mongo_conn_string = os.getenv('MONGO_DB_CONN_STRING')

    etl_pipeline = ETLPipeline(mongo_conn_string)
    etl_pipeline.fetch_clean_store('balance_sheet', 'ML_balancesheets')
    etl_pipeline.fetch_clean_store('cash_flow', 'ML_cashflows')
    etl_pipeline.fetch_clean_store('technicals', 'ML_incomestatements')
    etl_pipeline.fetch_clean_store('news_sentiment_data', 'ML_sentiment_data')