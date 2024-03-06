from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
from pymongo import MongoClient
import requests
import json

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MongoDB configuration
MONGO_DB_CONN_STRING = os.getenv("MONGO_CONN_STRING")
DB_NAME = "stock_data"

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure the API key is loaded correctly
if not OPENAI_API_KEY:
    logging.error("Failed to load OPENAI_API_KEY from environment variables.")
    exit(1)

# Initialize OpenAI client with the API key
client = OpenAI(api_key=OPENAI_API_KEY)

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

def fetch_selected_pairs():
    try:
        selected_pairs_collection = db['selected_pairs']
        selected_pairs = list(selected_pairs_collection.find())
        return selected_pairs
    except Exception as e:
        logging.error(f"An error occurred while fetching selected pairs from MongoDB: {e}")
        return []


def analyze_stock_data(stock_data):
    try:
        # Construct the messages for the chat
        messages = [
            {"role": "system",
             "content": "Your role is to act as a Quantitative Trading Engineer, providing expert advice and "
                        "solutions in the field of quantitative trading. Your goal is to deliver the most accurate "
                        "and logical information. Return your work in json. Keep your analysis high level and brief "
                        "limit context and prose. Limit your entire report to less than 150 words. Return just to "
                        "body of your presentation or memo without salutations or other formalities. You report will "
                        "have 3 sections: 1) Analysis: gives brief stance and history. 2) Numbers: This is a brief "
                        "overview of numbers that were used to make the decision based on technical indicators. 3) "
                        "Justification against the trade strategy which is this {trade_strategy}. Use language and "
                        "analogies for the 10th grade reading level so our users can understand. "},
            {"role": "user",
             "content": f"Analyze this stock data: {stock_data} and provide summary analysis as if presenting to "
                        f"investors with the managed portfolio getting updates about why certain investments were "
                        f"made automatically.  "}
        ]

        logging.info("Sending request to OpenAI API")

        # Call the OpenAI Chat API
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages
        )

        logging.info("Received response from OpenAI API")

        # Process the response
        analysis = response.choices[0].message.content
        logging.info(f"Usage: {response.usage}")

        return analysis

    except Exception as e:
        logging.error(f"An error occurred while analyzing stock data: {e}")


def send_teams_message(webhook_url, message):
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(webhook_url, headers=headers, data=json.dumps(message))
    print(response.headers)
    if response.status_code == 200:
        logging.info("Message sent to Teams successfully.")
    else:
        logging.error(f"Failed to send message to Teams. Status code: {response.text}")


def process_analysis_and_send_teams_message(analysis, teams_url):
    try:
        analysis_dict = json.loads(analysis)
        logging.info(f"Parsed analysis JSON: {analysis_dict}")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse analysis JSON: {e}")
        return

    message = {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": "0076D7",
        "summary": "Stock Analysis Summary",
        "sections": [{
            "activityTitle": "Stock Analysis Report",
            "activitySubtitle": "An overview of the analyzed stock data.",
            "facts": [],
            "markdown": True
        }]
    }

    facts = message["sections"][0]["facts"]

    # Directly use analysis content if 'overview' key is not present
    if "overview" in analysis_dict:
        overview = analysis_dict["overview"]
    else:
        overview = analysis  # Fallback to using the full analysis content

    facts.append({'name': "Overview", 'value': f"**{overview}**"})

    logging.info(f"Sending message to Teams: {message}")

    send_teams_message(teams_url, message)



if __name__ == "__main__":
    selected_pairs = fetch_selected_pairs()
    teams_url = 'https://data874.webhook.office.com/webhookb2/d308710c-ab52-4a02-8e6d-583b769a50dd@4f84582a-9476-452e-a8e6-0b57779f244f/IncomingWebhook/250f8a9f3c934f49b06e80202f6f09e9/6d2e1385-bdb7-4890-8bc5-f148052c9ef5'

    for pair in selected_pairs:
        stock_data_str = ', '.join([f"{key}: {value}" for key, value in pair.items()])
        analysis_result = analyze_stock_data(stock_data_str)  # This should return a JSON string
        if analysis_result:
            process_analysis_and_send_teams_message(analysis_result, teams_url)
