import pymongo
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from pymongo.mongo_client import MongoClient
from urllib.parse import quote_plus

def upload():
    username = quote_plus('Aloysius')
    password = quote_plus('AndricHemingway99!')
    
    # Connect to MongoDB
    uri = "mongodb+srv://"+username+":"+password+"@mnistwithmongodb.fromm.mongodb.net/?retryWrites=true&w=majority&appName=MNISTwithMongoDB"

    # Create a new client and connect to the server
    client = MongoClient(uri)
    db = client["mnist"]
    collection = db["images"]

    # Clear existing data
    collection.delete_many({})

    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Combine train and test data
    x_all = np.concatenate((x_train, x_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)

    # Upload data to MongoDB
    for i in range(len(x_all)):
        document = {
            "image": x_all[i].tolist(),  # Convert numpy array to list
            "label": int(y_all[i])
        }
        collection.insert_one(document)

    print(f"Uploaded {len(x_all)} MNIST images to MongoDB.")

if __name__ == "__main__":
    upload()