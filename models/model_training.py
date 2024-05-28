import os
import joblib
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Setup logging
logging.basicConfig(filename='model_training_logfile.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def get_latest_file(directory, file_prefix):
    """Get the latest file for a specific prefix based on version number."""
    files = os.listdir(directory)
    relevant_files = [f for f in files if f.startswith(file_prefix) and f.endswith(".pkl")]
    if relevant_files:
        latest_file = sorted(relevant_files, key=lambda x: int(x.split('_v')[-1].split('.')[0]), reverse=True)[0]
        return latest_file
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

def train_model(symbol, symbol_dir):
    """Train the model for a given symbol."""
    latest_model_file = get_latest_file(symbol_dir, f"model_{symbol}")
    latest_feature_scaler_file = get_latest_file(symbol_dir, f"feature_scaler_{symbol}")
    latest_target_scaler_file = get_latest_file(symbol_dir, f"target_scaler_{symbol}")
    latest_X_train_file = get_latest_file(symbol_dir, f"X_train_{symbol}")
    latest_X_test_file = get_latest_file(symbol_dir, f"X_test_{symbol}")
    latest_y_train_file = get_latest_file(symbol_dir, f"y_train_{symbol}")
    latest_y_test_file = get_latest_file(symbol_dir, f"y_test_{symbol}")

    if not latest_model_file or not latest_feature_scaler_file or not latest_target_scaler_file or \
       not latest_X_train_file or not latest_X_test_file or not latest_y_train_file or not latest_y_test_file:
        print(f"Missing files for {symbol}, skipping...")
        return

    model_path = os.path.join(symbol_dir, latest_model_file)
    feature_scaler_path = os.path.join(symbol_dir, latest_feature_scaler_file)
    target_scaler_path = os.path.join(symbol_dir, latest_target_scaler_file)
    X_train_path = os.path.join(symbol_dir, latest_X_train_file)
    X_test_path = os.path.join(symbol_dir, latest_X_test_file)
    y_train_path = os.path.join(symbol_dir, latest_y_train_file)
    y_test_path = os.path.join(symbol_dir, latest_y_test_file)

    # Load data files
    X_train = joblib.load(X_train_path)
    X_test = joblib.load(X_test_path)
    y_train = joblib.load(y_train_path)
    y_test = joblib.load(y_test_path)

    model = joblib.load(model_path)
    model.fit(X_train, y_train.ravel())

    mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"Model trained for {symbol} with MSE: {mse}")
    logging.info(f"Updated model for {symbol} with MSE: {mse}")

    # Save training metrics and metadata
    metadata = {
        "mse": mse,
        "model_version": latest_model_file.split('_')[-1].split('.pkl')[0],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "timestamp": logging.Formatter('%(asctime)s').format(logging.makeLogRecord({'levelname': 'INFO'}))
    }
    with open(os.path.join(symbol_dir, f"training_metadata_{symbol}.json"), 'w') as metafile:
        json.dump(metadata, metafile)

def process_models(base_dir):
    """Process models for each symbol found in the model directory."""
    symbols = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Found symbols: {symbols}")

    for symbol in symbols:
        symbol_dir = os.path.join(base_dir, symbol)
        print(f"Processing model for {symbol}")
        train_model(symbol, symbol_dir)

if __name__ == "__main__":
    model_dir = "modelfiles/"
    process_models(model_dir)