import pymongo
import numpy as np
from tensorflow import keras

# Load MNIST dataset from Keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["mnist_database"]
collection = db["images"]

# Function to insert data into MongoDB
def insert_data(images, labels, set_name):
    data = []
    for image, label in zip(images, labels):
        doc = {
            "image": image.tolist(),  # Convert numpy array to list
            "label": int(label),
            "set": set_name
        }
        data.append(doc)
    
    # Insert in batches for better performance
    batch_size = 1000
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        collection.insert_many(batch)

# Insert training data
print("Inserting training data...")
insert_data(x_train, y_train, "train")

# Insert test data
print("Inserting test data...")
insert_data(x_test, y_test, "test")

print("Database setup complete!")

# Verify the data
train_count = collection.count_documents({"set": "train"})
test_count = collection.count_documents({"set": "test"})

print(f"Number of training samples: {train_count}")
print(f"Number of test samples: {test_count}")