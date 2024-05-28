import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import pymongo
import hashlib


load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API')
MONGO_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')


def fetch_unique_symbols():
    client = pymongo.MongoClient(MONGO_CONN_STRING)
    try:
        db = client['stock_data']
        collection = db['selected_pairs']
        # Fetch unique symbols using the distinct method
        unique_symbols = collection.distinct('symbol')
        return unique_symbols
    finally:
        client.close()


def update_ticker_sentiment(filtered_data, company_ticker, ticker_data, ticker_cumulative_score):
    ticker_sentiment = filtered_data['ticker_sentiment'].get(company_ticker, {
        'cumulative_score': 0,
        'article_count': 0,
        'label_counts': {}
    })
    ticker_sentiment['cumulative_score'] += ticker_cumulative_score
    ticker_sentiment['article_count'] += 1
    ticker_sentiment_label = ticker_data.get('ticker_sentiment_label', 'Neutral')
    ticker_sentiment['label_counts'][ticker_sentiment_label] = ticker_sentiment['label_counts'].get(ticker_sentiment_label, 0) + 1
    filtered_data['ticker_sentiment'][company_ticker] = ticker_sentiment


def process_source_and_topic_sentiment(filtered_data, article):
    source = article.get('source', 'Unknown')
    sentiment_score = article.get('overall_sentiment_score', 0)

    # Update source sentiment
    source_sentiment = filtered_data['sentiment_by_source'].get(source, {
        'cumulative_score': 0,
        'article_count': 0
    })
    source_sentiment['cumulative_score'] += sentiment_score
    source_sentiment['article_count'] += 1
    filtered_data['sentiment_by_source'][source] = source_sentiment

    # Update topic sentiment
    for topic_data in article.get('topics', []):
        topic = topic_data.get('topic', 'Unknown')
        relevance_score = float(topic_data.get('relevance_score', 0))
        topic_sentiment = filtered_data['sentiment_by_topic'].get(topic, {
            'cumulative_score': 0,
            'topic_relevance': 0
        })
        topic_sentiment['cumulative_score'] += sentiment_score * relevance_score
        topic_sentiment['topic_relevance'] += relevance_score
        filtered_data['sentiment_by_topic'][topic] = topic_sentiment


def finalize_filtered_data(filtered_data, total_articles, cumulative_sentiment_score, sentiment_label_counts):
    if total_articles > 0:
        # Update overall sentiment
        filtered_data['overall_sentiment']['average_score'] = cumulative_sentiment_score / total_articles
        filtered_data['overall_sentiment']['label_percentages'] = {label: count / total_articles for label, count in sentiment_label_counts.items()}

        # Update ticker sentiment averages and percentages
        for ticker, ticker_data in filtered_data['ticker_sentiment'].items():
            ticker_data['average_score'] = ticker_data['cumulative_score'] / ticker_data['article_count']
            ticker_data['label_percentages'] = {label: count / ticker_data['article_count'] for label, count in ticker_data['label_counts'].items()}

        # Update source and topic averages
        for source_data in filtered_data['sentiment_by_source'].values():
            source_data['average_score'] = source_data['cumulative_score'] / source_data['article_count']
        for topic_data in filtered_data['sentiment_by_topic'].values():
            topic_data['weighted_average_score'] = topic_data['cumulative_score'] / topic_data['topic_relevance']



def fetch_news_sentiment_data(ticker, api_key):
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}")
        return None


def generate_data_signature(sentiment_data):
    # Safely access 'average_score' with a default value of 0 if not found
    overall_average_score = sentiment_data.get('overall_sentiment', {}).get('average_score', 0)

    # Safely get the symbol from sentiment_data with a default value of an empty string
    symbol = sentiment_data.get('symbol', '')

    # Safely access 'cumulative_score' for the symbol in 'ticker_sentiment' with a default value of 0
    ticker_cumulative_score = sentiment_data.get('ticker_sentiment', {}).get(symbol, {}).get('cumulative_score', 0)

    # Generate a string representation of the key sentiment metrics
    signature_str = f"{overall_average_score}_{ticker_cumulative_score}_{len(sentiment_data.get('sentiment_by_source', {}))}_{len(sentiment_data.get('sentiment_by_topic', {}))}"

    # Use hashlib to generate a hash of the signature string
    return hashlib.sha256(signature_str.encode()).hexdigest()


def store_sentiment_data_in_mongo(filtered_sentiment, company_ticker):
    client = pymongo.MongoClient(MONGO_CONN_STRING)
    try:
        db = client['stock_data']
        collection = db['news_sentiment_data']

        # Generate a signature for the current sentiment data
        current_signature = generate_data_signature(filtered_sentiment)

        # Attempt to find the most recent record for the same symbol
        existing_record = collection.find_one({'symbol': company_ticker},
                                              sort=[('datetime_imported', pymongo.DESCENDING)])

        if existing_record:
            # Generate a signature for the existing record
            existing_signature = generate_data_signature(existing_record)

            # Compare the signatures to determine if the data has materially changed
            if current_signature != existing_signature:
                collection.insert_one(filtered_sentiment)
                print(f"New sentiment data for {company_ticker} stored in MongoDB.")
            else:
                print(f"No significant change in sentiment data for {company_ticker}. No new record inserted.")
        else:
            # No existing record for this symbol, so insert the new data
            collection.insert_one(filtered_sentiment)
            print(f"Data for {company_ticker} stored in MongoDB.")
    finally:
        client.close()


def filter_sentiment_data(sentiment_data, company_ticker, start_date, end_date):
    # Convert start_date and end_date to datetime objects for comparison
    start_date_dt = datetime.strptime(start_date, "%Y%m%d")
    end_date_dt = datetime.strptime(end_date, "%Y%m%d")

    filtered_data = {
        'overall_sentiment': {'average_score': 0, 'label_percentages': {}},
        'ticker_sentiment': {},
        'sentiment_by_source': {},
        'sentiment_by_topic': {},
        'datetime_imported': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        'symbol': company_ticker
    }

    # Check if 'feed' key exists in sentiment_data and it's not empty
    if 'feed' not in sentiment_data or not sentiment_data['feed']:
        print(f"No news feed data found for {company_ticker} within the specified date range.")
        return filtered_data

    total_articles, cumulative_sentiment_score = 0, 0
    sentiment_label_counts = {}

    for article in sentiment_data['feed']:
        # Verify article date is within the desired range
        article_date_dt = datetime.strptime(article.get('time_published', '')[:8], "%Y%m%d")
        if start_date_dt <= article_date_dt <= end_date_dt:
            total_articles += 1
            sentiment_score = article.get('overall_sentiment_score', 0)
            sentiment_label = article.get('overall_sentiment_label', 'Neutral')
            cumulative_sentiment_score += sentiment_score
            sentiment_label_counts[sentiment_label] = sentiment_label_counts.get(sentiment_label, 0) + 1

            # Process ticker sentiment
            for ticker_data in article.get('ticker_sentiment', []):
                if ticker_data.get('ticker') == company_ticker:
                    # Ensure numeric values for calculation
                    ticker_sentiment_score = float(ticker_data.get('ticker_sentiment_score', 0))
                    relevance_score = float(ticker_data.get('relevance_score', 1))
                    ticker_cumulative_score = ticker_sentiment_score * relevance_score
                    update_ticker_sentiment(filtered_data, company_ticker, ticker_data, ticker_cumulative_score)

            # Process sentiment by source and topic
            process_source_and_topic_sentiment(filtered_data, article)

    # Final calculations and updates
    finalize_filtered_data(filtered_data, total_articles, cumulative_sentiment_score, sentiment_label_counts)

    return filtered_data


def fetch_and_store_sentiment(ticker, start_date, end_date, api_key):
    sentiment_data = fetch_news_sentiment_data(ticker, api_key)
    if sentiment_data:
        filtered_sentiment = filter_sentiment_data(sentiment_data, ticker, start_date, end_date)
        store_sentiment_data_in_mongo(filtered_sentiment, ticker)
        return filtered_sentiment
    else:
        print("Failed to fetch sentiment data.")
        return None


def main():
    # Assume start_date is dynamically set to one week before today, and end_date is set to today
    start_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")
    end_date = datetime.now(timezone.utc).strftime("%Y%m%d")

    # Fetch unique symbols from MongoDB
    unique_symbols = fetch_unique_symbols()

    # Iterate over each symbol and fetch/store sentiment data
    for ticker in unique_symbols:
        print(f"Processing sentiment data for {ticker}...")
        filtered_sentiment = fetch_and_store_sentiment(ticker, start_date, end_date, ALPHA_VANTAGE_API_KEY)
        if filtered_sentiment:
            # If you want to print or process the filtered sentiment data for each ticker, do it here
            pass

# if filtered_sentiment:
#     # Print the filtered sentiment data
#     print(filtered_sentiment)

if __name__ == "__main__":
    main()

