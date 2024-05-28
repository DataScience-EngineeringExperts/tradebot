import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras import optimizers
from scikeras.wrappers import KerasRegressor
import numpy as np
from datetime import datetime
from tensorflow.keras import regularizers
import joblib
from joblib import dump
import os
import logging
from sys import argv

# Initialize logging
logging.basicConfig(filename='logfile.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Specify the cgroup path and the memory limit
CGROUP_PATH = "/sys/fs/cgroup/tradebot"
MEMORY_LIMIT = "42949672960"  # 40G in bytes

try:
    # Set the memory limit
    with open(os.path.join(CGROUP_PATH, "memory.max"), "w") as f:
        f.write(MEMORY_LIMIT)
    logging.info(f"Set memory limit to {int(MEMORY_LIMIT) / (1024 * 1024 * 1024)} GB")

    # Add the script's PID to the cgroup
    pid = os.getpid()
    with open(os.path.join(CGROUP_PATH, "cgroup.procs"), "w") as f:
        f.write(str(pid))
    logging.info(f"Added PID {pid} to cgroup")

except Exception as e:
    logging.error("Failed to modify cgroup: " + str(e))


# Setup logging
logging.basicConfig(filename='logfile.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

"""
Stock Price Prediction Script

This script loads financial data from a MongoDB database, prepares the data for training a neural network model,
trains the model to predict the stock price 180 days into the future, and evaluates the model's performance.

Main execution:
   - Retrieves a list of all collections in the MongoDB database that contain processed data.
   - Optionally limits the processing to the first collection for development purposes.
   - Iterates over each collection:
     - Loads the data from the collection using `load_data`.
     - Prepares the data for training using `prepare_data`.
     - Performs hyperparameter optimization using `hyperparameter_optimization`.
     - Builds the neural network model using `build_model` with the best hyperparameters.
     - Trains and evaluates the model using `train_and_evaluate_model`.
     - Stores the predicted prices and datetime of calculation in the `results` list.
     - Prints the training status and evaluation metrics for each collection.

The script utilizes the following libraries:
- os: For loading environment variables.
- pandas: For data manipulation and analysis.
- pymongo: For interacting with the MongoDB database.
- dotenv: For loading environment variables from a .env file.
- scikit-learn: For data preprocessing, splitting, and hyperparameter optimization.
- TensorFlow and Keras: For building and training the neural network model.
- scikeras: For integrating Keras models with scikit-learn.

"""


# Load environment variables
load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Setup MongoDB connection
client = MongoClient(MONGO_DB_CONN_STRING)
db = client['machinelearning']


def load_data(collection_name):
    """Load data from MongoDB collection into DataFrame, ensure data types are numeric, and handle missing values."""
    collection = db[collection_name]
    data = pd.DataFrame(list(collection.find()))

    # Handling date fields correctly
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'].apply(lambda x: x.get('$date') if isinstance(x, dict) else x))
    if 'date_imported' in data.columns:
        data['date_imported'] = pd.to_datetime(data['date_imported'].apply(lambda x: x.get('$date') if isinstance(x, dict) else x))

    # Flatten nested structures and extract nested data
    for col in data.columns:
        if isinstance(data[col].iloc[0], dict):
            nested_df = pd.json_normalize(data[col])
            nested_df.columns = [f"{col}_{subkey}" for subkey in nested_df.columns]
            data = pd.concat([data.drop(columns=[col]), nested_df], axis=1)

    # Convert all columns to numeric where possible, setting errors to coerce
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Fill missing values with zeros
    data.fillna(0, inplace=True)

    print(f"Data types after conversion:\n{data.dtypes}")
    return data


def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    ''' Wrapper function to create a LearningRateScheduler with step decay schedule. '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))
    return LearningRateScheduler(schedule)


def prepare_data(df, symbol, model_dir):
    """Prepare data for model training by scaling features, selecting relevant features, and splitting into train/test sets."""
    if df.empty:
        print("No data to prepare.")
        return None, None, None, None, None, None, None

    try:
        # Ensure 'close' is numeric and handle missing values
        df['close'] = pd.to_numeric(df['close'], errors='coerce').fillna(0)

        # Identify feature columns excluding known non-feature columns
        feature_columns = [col for col in df.columns if col not in ['close', 'symbol', 'date', 'date_imported']]

        # Scale features
        feature_scaler = StandardScaler()
        features_scaled = feature_scaler.fit_transform(df[feature_columns])

        # Scale the target
        target_scaler = StandardScaler()
        target_scaled = target_scaler.fit_transform(df[['close']].values)  # Ensure 2D input for scaler

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.1, random_state=42)

        print("Data preparation successful.")
        return X_train, X_test, y_train, y_test, feature_scaler, target_scaler, feature_columns

    except Exception as e:
        print(f"Error preparing data: {e}")
        return None, None, None, None, None, None, None


def build_model(input_dim, optimizer='adam', activation='relu', learning_rate=0.001, dropout_rate=0.2,
                l2_regularization=0.01, num_layers=3, units_per_layer=128):
    """Define and compile a neural network model with dropout and L2 regularization"""
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for _ in range(num_layers):
        model.add(Dense(units_per_layer, activation=activation, kernel_regularizer=regularizers.l2(l2_regularization)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    if optimizer == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    model.compile(optimizer=opt, loss='mean_squared_error',
                  metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])
    print("Model built and compiled.")
    return model


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, feature_scaler, target_scaler, symbol):
    """Train the model using K-Fold cross-validation and evaluate its performance on a test set."""
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Reduced number of splits for quicker evaluations
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)  # Reduced patience for quicker convergence
    lr_scheduler = step_decay_schedule(initial_lr=0.005, decay_factor=0.5, step_size=20)  # Adjusted step size for more frequent updates

    cv_scores = []
    for train_index, val_index in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Train the model on the training fold
        model.fit(X_train_fold, y_train_fold, epochs=30, batch_size=64,
                  validation_data=(X_val_fold, y_val_fold),
                  callbacks=[early_stop, lr_scheduler], verbose=1)

        # Evaluate the model on the validation fold
        scores = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        cv_scores.append(scores)

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation MAE: {np.mean([score[1] for score in cv_scores])}")
    print(f"Mean cross-validation MAPE: {np.mean([score[2] for score in cv_scores])}")

    # Evaluate the model on the test set
    test_scores = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test set scores: {test_scores}")
    logging.info(f"Test set scores: {test_scores}")

    # Save the model and scalers with versioning and cleanup
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model_and_scalers_with_cleanup(model, feature_scaler, target_scaler, symbol, version)

    return model


def hyperparameter_optimization(X_train, y_train):
    """Perform hyperparameter optimization to find the best model configurations."""
    # param_grid = {
    #     'model__optimizer': ['adam', 'rmsprop', 'sgd'],
    #     'model__activation': ['relu', 'elu', 'leaky_relu'],
    #     'model__learning_rate': [0.01, 0.001, 0.0001],
    #     'model__dropout_rate': [0.1, 0.2, 0.3],
    #     'model__l2_regularization': [0.01, 0.001, 0.0001],
    #     'batch_size': [16, 32, 64],
    #     'epochs': [35, 100, 125],
    #     'model__num_layers': [3, 4, 5],  # Adjust based on the complexity needed
    #     'model__units_per_layer': [64, 128, 256]  # Units per hidden layer
    # }

    param_grid = {
        'model__optimizer': ['adam', 'rmsprop', 'sgd'],
        'model__activation': ['relu', 'elu', 'leaky_relu'],
        'model__learning_rate': [0.01, 0.001, 0.0001],
        'model__dropout_rate': [0.1],
        'model__l2_regularization': [0.01],
        'batch_size': [64],
        'epochs': [35],
        'model__num_layers': [3],  # Adjust based on the complexity needed
        'model__units_per_layer': [64]  # Units per hidden layer
    }

    def create_model(optimizer='adam', activation='relu', learning_rate=0.001, dropout_rate=0.2, l2_regularization=0.01,
                     num_layers=3, units_per_layer=128):
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
        for _ in range(num_layers):
            model.add(
                Dense(units_per_layer, activation=activation, kernel_regularizer=regularizers.l2(l2_regularization)))
            model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        opt = optimizers.get(optimizer)
        opt.learning_rate = learning_rate
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_absolute_error'])
        return model


    model = KerasRegressor(model=create_model, verbose=0)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    print(f"Optimal parameters found: {grid_search.best_params_}")
    print(f"Best negative mean squared error: {grid_search.best_score_}")

    return grid_search.best_params_



def evaluate_and_predict(model, X_test, y_test, target_scaler):
    """Evaluate the trained model on the test set and provide scaled predictions."""
    try:
        # Evaluate model performance
        metrics = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test loss: {metrics[0]}, Test MAE: {metrics[1]}, Test MAPE: {metrics[2]}")

        # Generate predictions
        predictions = model.predict(X_test).flatten()

        # Rescale predictions to the original scale
        rescaled_predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        # Print a sample of predictions for verification
        print(f"Sample of rescaled predictions: {rescaled_predictions[:5]}")

        return rescaled_predictions, metrics
    except Exception as e:
        print(f"Error during model evaluation and prediction: {e}")
        return None, None


def get_next_version_number(symbol, model_dir):
    """Retrieve the next version number for a given symbol by checking existing files."""
    files = os.listdir(model_dir)
    versions = [
        int(f.split('_v')[-1].split('.')[0]) for f in files
        if f.startswith(f"model_{symbol}_v") and f.endswith(".pkl")
    ]
    return max(versions, default=0) + 1


def save_scalers(feature_scaler, target_scaler, path='scalers/'):
    joblib.dump(feature_scaler, os.path.join(path, 'feature_scaler.gz'))
    joblib.dump(target_scaler, os.path.join(path, 'target_scaler.gz'))

def load_scalers(path='scalers/'):
    feature_scaler = joblib.load(os.path.join(path, 'feature_scaler.gz'))
    target_scaler = joblib.load(os.path.join(path, 'target_scaler.gz'))
    return feature_scaler, target_scaler


def store_predictions(predictions, symbol, collection, target_scaler):
    # Rescale predictions to the original scale
    predictions_rescaled = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # Convert numpy types to Python types
    predictions_rescaled = convert_numpy_to_python(predictions_rescaled)

    prediction_data = [{
        'symbol': symbol,
        'predicted_price': pred,
        'datetime_calculated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    } for pred in predictions_rescaled]
    collection.insert_many(prediction_data)

    # Log a sample of stored predictions
    log_sample_predictions(collection)


def log_sample_predictions(collection):
    stored_results = pd.DataFrame(list(collection.find()))
    logging.info("Sample of stored predictions:")
    logging.info(stored_results[['symbol', 'predicted_price', 'datetime_calculated']].head())
    print(stored_results[['symbol', 'predicted_price', 'datetime_calculated']].head())


def save_model_and_scalers(model, feature_scaler, target_scaler, symbol, model_dir, X_train, X_test, y_train, y_test):
    """
    Save the model, scalers, and data with a sequential version number.
    """
    version = get_next_version_number(symbol, model_dir)
    model_path = os.path.join(model_dir, f"model_{symbol}_v{version}.pkl")
    feature_scaler_path = os.path.join(model_dir, f"feature_scaler_{symbol}_v{version}.pkl")
    target_scaler_path = os.path.join(model_dir, f"target_scaler_{symbol}_v{version}.pkl")
    X_train_path = os.path.join(model_dir, f"X_train_{symbol}_v{version}.pkl")
    X_test_path = os.path.join(model_dir, f"X_test_{symbol}_v{version}.pkl")
    y_train_path = os.path.join(model_dir, f"y_train_{symbol}_v{version}.pkl")
    y_test_path = os.path.join(model_dir, f"y_test_{symbol}_v{version}.pkl")

    joblib.dump(model, model_path)
    joblib.dump(feature_scaler, feature_scaler_path)
    joblib.dump(target_scaler, target_scaler_path)
    joblib.dump(X_train, X_train_path)
    joblib.dump(X_test, X_test_path)
    joblib.dump(y_train, y_train_path)
    joblib.dump(y_test, y_test_path)

    logging.info(f"Model saved to {model_path}")
    logging.info(f"Feature scaler saved to {feature_scaler_path}")
    logging.info(f"Target scaler saved to {target_scaler_path}")
    logging.info(f"X_train saved to {X_train_path}")
    logging.info(f"X_test saved to {X_test_path}")
    logging.info(f"y_train saved to {y_train_path}")
    logging.info(f"y_test saved to {y_test_path}")



def create_symbol_directory(symbol, parent_dir='modelfiles'):
    """Create a separate directory for each symbol within the parent directory."""
    symbol_dir = os.path.join(parent_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    return symbol_dir


def cleanup_old_files(symbol_dir, max_files=5):
    """Remove old model files based on modification time, keeping only the most recent ones."""
    files = glob.glob(os.path.join(symbol_dir, '*'))
    files.sort(key=os.path.getmtime, reverse=True)

    if len(files) > max_files:
        old_files = files[max_files:]
        for file in old_files:
            os.remove(file)
            logging.info(f"Removed old file: {file}")


def save_model_and_scalers_with_cleanup(model, feature_scaler, target_scaler, symbol, version):
    """
    Save the model and scalers with unique filenames, versioning, and perform cleanup of old files.

    Args:
        model: The trained Keras model.
        feature_scaler: The fitted feature scaler object.
        target_scaler: The fitted target scaler object.
        symbol: The stock symbol associated with the model.
        version: The version number or timestamp of the model.
    """
    model_dir = "modelfiles"
    os.makedirs(model_dir, exist_ok=True)

    # Construct file paths
    model_path = os.path.join(model_dir, f"model_{symbol}_{version}.pkl")
    feature_scaler_path = os.path.join(model_dir, f"feature_scaler_{symbol}_{version}.pkl")
    target_scaler_path = os.path.join(model_dir, f"target_scaler_{symbol}_{version}.pkl")

    # Save the model and scalers using joblib
    dump(model, model_path)
    dump(feature_scaler, feature_scaler_path)
    dump(target_scaler, target_scaler_path)


def convert_numpy_to_python(data):
    """
    Convert numpy types in the data to native Python types for serialization compatibility.
    """
    if isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_numpy_to_python(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_python(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_numpy_to_python(item) for item in data)
    else:
        return data


def main(limit_collections=True):
    try:
        model_outputs = db['model_outputs']
        collection_names = [name for name in db.list_collection_names() if 'processed_data' in name]

        # if limit_collections:
        #     collection_names = collection_names[:1]  # Limit to one collection for initial testing

        for collection_name in collection_names:
            logging.info(f"Processing {collection_name}...")
            print(f"Processing {collection_name}...")
            symbol = collection_name.split('_')[0]

            try:
                data = load_data(collection_name)
                if data is not None:
                    model_dir = create_symbol_directory(symbol)  # Create a directory for the symbol
                    X_train, X_test, y_train, y_test, feature_scaler, target_scaler, data_paths = prepare_data(data, symbol, model_dir)
                    if X_train is not None:
                        best_params = hyperparameter_optimization(X_train, y_train)
                        model_params = {k.replace('model__', ''): v for k, v in best_params.items() if k.startswith('model__')}
                        model = build_model(X_train.shape[1], **model_params)
                        model = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, feature_scaler,
                                                         target_scaler, symbol)
                        predictions, metrics = evaluate_and_predict(model, X_test, y_test, target_scaler)
                        store_predictions(predictions, symbol, model_outputs, target_scaler)

                        # Save the model, scalers, and data
                        save_model_and_scalers(model, feature_scaler, target_scaler, symbol, model_dir, X_train, X_test,
                                               y_train, y_test)
                    else:
                        logging.error("Failed to prepare data for training.")
                        print(f"Failed to prepare data for {symbol}")
                else:
                    logging.error("Data loading failed.")
                    print(f"Data load failed {symbol}")
            except Exception as e:
                logging.error(f"Error processing {collection_name}: {e}")
    except Exception as e:
        logging.error(f"An error occurred in main execution: {e}")


if __name__ == "__main__":
    logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s', handlers=[logging.FileHandler('logfile.log'), logging.StreamHandler()])
    # limit = '--all' not in argv
    main()