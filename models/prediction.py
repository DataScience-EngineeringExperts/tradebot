import sys
import os
import joblib
import logging
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Adjust system path to include the directory where the teams_communicator module is located
script_dir = os.path.dirname(__file__)  # Gets the directory where the current script is located
sys.path.append(os.path.join(script_dir, '..'))  # Appends the path to the module

from teams_communicator import TeamsCommunicator  # Now we can import TeamsCommunicator

# Load environment variables
load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Setup MongoDB connection
client = MongoClient(MONGO_DB_CONN_STRING)
db_ml = client['machinelearning']
db_predictions = client['predictions']

# Setup logging
logging.basicConfig(filename='prediction_logfile.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def get_latest_file(directory, file_type):
    """Get the latest file for a specific type based on version number."""
    try:
        files = os.listdir(directory)
        relevant_files = [f for f in files if f.startswith(file_type) and f.endswith(".pkl")]
        if not relevant_files:
            logging.warning(f"No files of type '{file_type}' found in directory: {directory}")
            return None  # No files found
        latest_file = sorted(relevant_files, key=lambda x: int(x.split('_v')[-1].split('.')[0]), reverse=True)[0]
        return os.path.join(directory, latest_file)  # Return the full path
    except Exception as e:
        logging.error(f"Failed to find the latest file in {directory}: {str(e)}")
        return None


def load_and_verify_scalers(feature_scaler_path, target_scaler_path):
    """Load and verify the feature and target scalers."""
    try:
        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        return feature_scaler, target_scaler
    except Exception as e:
        logging.error(f"Error loading scaler files: {str(e)}")
        return None, None

def predict_new_prices(symbol, model_dir, new_data):
    """Predict prices using the trained model and new input data."""
    symbol_dir = os.path.join(model_dir, symbol)
    model_path = get_latest_file(symbol_dir, 'model')
    feature_scaler_path = get_latest_file(symbol_dir, 'feature_scaler')
    target_scaler_path = get_latest_file(symbol_dir, 'target_scaler')

    if not model_path or not feature_scaler_path or not target_scaler_path:
        logging.error(f"Required files are missing for symbol {symbol} in directory {symbol_dir}")
        return

    model = joblib.load(model_path)
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)

    # Process features as before
    excluded_columns = ['close', 'symbol', 'date', 'date_imported']
    features = new_data.drop(columns=excluded_columns, errors='ignore')
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
    features_scaled = feature_scaler.transform(features)

    # Predict scaled prices and inverse transform to original scale
    predicted_scaled = model.predict(features_scaled)
    predicted_price = target_scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()[-1]

    # Prepare metadata and store results
    prediction_date = pd.to_datetime(new_data['date'].max()) + timedelta(days=180)
    save_prediction(symbol, prediction_date, predicted_price)


def save_prediction(symbol, prediction_date, predicted_price):
    """Save prediction results to MongoDB with metadata if it doesn't already exist."""
    current_month_year = datetime.now().strftime("%B_%Y")  # Use the current month and year for the collection name
    collection = db_predictions[current_month_year]  # Collection named after the current month and year

    # Ensure that the predicted_price is a native Python float, not a numpy float
    predicted_price = float(predicted_price)

    # Check if a record with the same symbol, prediction_date, and predicted_price already exists
    existing_record = collection.find_one({
        'symbol': symbol,
        'prediction_date': prediction_date,
        'predicted_price': predicted_price
    })

    if existing_record:
        logging.info(f"Prediction for {symbol} on {prediction_date.strftime('%Y-%m-%d')} with price {predicted_price} already exists. Skipping insertion.")
    else:
        # Metadata and prediction details
        data_entry = {
            'symbol': symbol,
            'prediction_date': prediction_date,
            'predicted_price': predicted_price,
            'entry_date': datetime.now()
        }

        # Insert data entry into the database
        collection.insert_one(data_entry)
        logging.info(f"Inserted prediction for {symbol} on {prediction_date.strftime('%Y-%m-%d')}: {predicted_price}")



def process_predictions(model_dir):
    """Process predictions for each symbol found in the MongoDB collection."""
    symbols = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    for symbol in db.list_collection_names():
        cleaned_symbol = symbol.replace('_processed_data', '')
        if cleaned_symbol in symbols:
            collection = db[symbol]
            documents = collection.find({})
            all_data = []

            for document in documents:
                # Assuming document contains a dictionary that can be directly appended
                all_data.append(document)

            if all_data:
                new_data = pd.DataFrame(all_data)
                new_data['date'] = pd.to_datetime(new_data['date'])  # Convert date column to datetime
                prediction_date, predicted_price = predict_new_prices(cleaned_symbol, model_dir, new_data)
                logging.info(f"Predicted price for {cleaned_symbol} on {prediction_date}: {predicted_price}")
            else:
                logging.info(f"No data found for {cleaned_symbol}")
        else:
            logging.info(f"No model directory found for {cleaned_symbol}")


if __name__ == "__main__":
    model_dir = "modelfiles/"
    if not os.path.exists(model_dir):
        logging.error(f"Model directory {model_dir} does not exist.")
        print(f"Model directory {model_dir} does not exist.")
    else:
        communicator = TeamsCommunicator("ml_database", "prediction_logs")  # Initialize TeamsCommunicator
        symbols = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        for symbol in db_ml.list_collection_names():
            cleaned_symbol = symbol.replace('_processed_data', '')
            if cleaned_symbol in symbols:
                collection = db_ml[symbol]
                documents = collection.find({})
                all_data = pd.DataFrame(list(documents))
                if not all_data.empty:
                    all_data['date'] = pd.to_datetime(all_data['date'])
                    predict_new_prices(cleaned_symbol, model_dir, all_data)
                else:
                    logging.info(f"No data found for {cleaned_symbol}")
            else:
                logging.info(f"No model directory found for {cleaned_symbol}")
