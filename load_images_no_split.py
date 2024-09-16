import pymongo
import numpy as np
from tensorflow import keras

# Load MNIST dataset from Keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Combine train and test data
x_all = np.concatenate((x_train, x_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["mnist_database_no_split"]
collection = db["images"]

# Function to insert data into MongoDB
def insert_data(images, labels):
    data = []
    for image, label in zip(images, labels):
        doc = {
            "image": image.tolist(),  # Convert numpy array to list
            "label": int(label)
        }
        data.append(doc)
    
    # Insert in batches for better performance
    batch_size = 1000
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        collection.insert_many(batch)

# Insert all data
print("Inserting MNIST data...")
insert_data(x_all, y_all)

print("Database setup complete!")

# Verify the data
total_count = collection.count_documents({})

print(f"Total number of samples: {total_count}")