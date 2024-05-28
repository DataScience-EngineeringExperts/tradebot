import openai
import os
from dotenv import load_dotenv
import logging
from pymongo import MongoClient
import json
from openai import OpenAI
from datetime import datetime, timedelta

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MongoDB configuration
MONGO_DB_CONN_STRING = os.getenv("MONGO_DB_CONN_STRING")
DB_NAME = "stock_data"

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure the API key is loaded correctly
if not OPENAI_API_KEY:
    logging.error("Failed to load OPENAI_API_KEY from environment variables.")
    exit(1)

# Initialize OpenAI client with the API key
openai.api_key = OPENAI_API_KEY

client = OpenAI()

# MongoDB connection
mongo_client = MongoClient(MONGO_DB_CONN_STRING)
db = mongo_client[DB_NAME]

trade_stats = {
    """
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
    """
}

# Read trading strategy from file
def read_trading_strategy(file_path):
    try:
        with open(file_path, 'r') as file:
            strategy = file.read()
        return strategy
    except Exception as e:
        logging.error(f"An error occurred while reading the trading strategy file: {e}")
        return ""

trading_strategy = read_trading_strategy('trading_strategy.md')

def fetch_selected_pairs():
    try:
        selected_pairs_collection = db['selected_pairs']
        selected_pairs = list(selected_pairs_collection.find())
        return selected_pairs
    except Exception as e:
        logging.error(f"An error occurred while fetching selected pairs from MongoDB: {e}")
        return []

def insert_analysis_result(db, analysis_result):
    try:
        fourteen_days_ago = datetime.now() - timedelta(days=14)
        existing_document = db.openai_analysis.find_one({
            "symbol": analysis_result["symbol"],
            "datetime": {"$gte": fourteen_days_ago.isoformat()}
        })

        if existing_document:
            db.openai_analysis.delete_one({"_id": existing_document["_id"]})
            logging.info(f"Deleted old document for {analysis_result['symbol']}.")

        db.openai_analysis.insert_one(analysis_result)
        logging.info(f"Successfully inserted analysis result for {analysis_result['symbol']}")
    except Exception as e:
        logging.error(f"An error occurred while inserting the analysis result: {e}")

# send to open ai for analysis using prompt specific data
def analyze_stock_data(stock_data):
    try:
        fourteen_days_ago = datetime.now() - timedelta(days=14)
        existing_document = db.openai_analysis.find_one({
            "symbol": stock_data["symbol"],
            "datetime": {"$gte": fourteen_days_ago.isoformat()}
        })

        if existing_document:
            logging.info(f"Document for {stock_data['symbol']} within the last 14 days already exists. Skipping analysis.")
            return

        messages = [
            {"role": "system",
             "content": "Your role is to act as a Quantitative Trading Engineer, providing expert advice and "
                        "solutions in the field of quantitative trading. Your goal is to deliver the most accurate "
                        "and logical information. Return your work in json. Keep your analysis high level and brief "
                        "limit context and prose. Limit your entire report to less than 150 words. Return just the "
                        "body of your presentation or memo without salutations or other formalities. Your report will "
                        "have 3 sections: 1) Analysis: gives brief stance and history. 2) Numbers: This is a brief "
                        "overview of numbers that were used to make the decision based on technical indicators. 3) "
                        f"Justification against the trade strategy which is this {trading_strategy}. Use language and "
                        "analogies for the 10th grade reading level so our users can understand."},
            {"role": "user",
             "content": f"Analyze this stock data: {stock_data} and provide summary analysis as if presenting to "
                        "investors with the managed portfolio getting updates about why certain investments were "
                        "made automatically."}
        ]

        logging.info("Sending request to OpenAI API")

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0
        )

        logging.info("Received response from OpenAI API")

        analysis_content = completion.choices[0].message.content
        logging.info(f"Usage: {completion.usage}")

        analysis_result = {
            "symbol": stock_data["symbol"],
            "datetime": datetime.now().isoformat(),
            "content": analysis_content,
            "token_usage": {
                "completion_tokens": completion.usage.completion_tokens,
                "prompt_tokens": completion.usage.prompt_tokens,
                "total_tokens": completion.usage.total_tokens
            }
        }

        # Insert the analysis result into MongoDB
        insert_analysis_result(db, analysis_result)

        # Print the analysis result as JSON
        print(json.dumps(analysis_result, default=str, indent=2))

        return analysis_content

    except Exception as e:
        logging.error(f"An error occurred while analyzing stock data: {e}")

if __name__ == "__main__":
    selected_pairs = fetch_selected_pairs()

    for pair in selected_pairs:
        analyze_stock_data(pair)  # This should process and log the analysis result
