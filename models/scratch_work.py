from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_DB_CONN_STRING = os.getenv('MONGO_DB_CONN_STRING')

# Setup MongoDB connection
client = MongoClient(MONGO_DB_CONN_STRING)
db = client['cleaneddata_ml']

def get_sample_data(collection_name, sample_size=1):
    """Retrieve sample data from the specified MongoDB collection."""
    collection = db[collection_name]
    cursor = collection.find().limit(sample_size)
    data = [doc for doc in cursor]
    return data

# Example usage
sample_data = get_sample_data('AGCO_processed_data')
for doc in sample_data:
    print(doc)
