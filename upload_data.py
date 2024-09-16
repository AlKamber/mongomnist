import pymongo
import numpy as np
from keras.datasets import mnist

def upload():
    # Connect to MongoDB
    client = pymongo.MongoClient("mongodb://localhost:27017/")
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