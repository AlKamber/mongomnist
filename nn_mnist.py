import pymongo
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["mnist_database"]
collection = db["images"]

# Function to fetch data from MongoDB
def fetch_data_from_mongodb(set_name):
    data = []
    labels = []
    for doc in collection.find({"set": set_name}):
        data.append(np.array(doc["image"]))
        labels.append(doc["label"])
    return np.array(data), np.array(labels)

# Fetch training and test data
print("Fetching training data...")
x_train, y_train = fetch_data_from_mongodb("train")
print("Fetching test data...")
x_test, y_test = fetch_data_from_mongodb("test")

# Normalize pixel values
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Define the model
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
print("Training the model...")
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate the model
print("Evaluating the model...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")