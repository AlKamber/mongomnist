import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
import warnings
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from pymongo.mongo_client import MongoClient
from urllib.parse import quote_plus

# Suppress specific Keras warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')


def get_mongo_client():
    username = quote_plus('Aloysius')
    password = quote_plus('msuOvojkA2VtCiL2')
    
    # Connect to MongoDB
    uri = "mongodb+srv://"+username+":"+password+"@mnistwithmongodb.fromm.mongodb.net/?retryWrites=true&w=majority&appName=MNISTwithMongoDB"

    client = MongoClient(uri)
    return client


# function to get the data from mongoDB
def fetch_data(client=None):
    counter = 0
    print("Fetching data...")
    if client is None:
        client = get_mongo_client()
    db = client.mnist
    collection = db.images
    X = []
    y = []
    for doc in collection.find():
        X.append(doc['image'])
        y.append(doc['label'])
        counter += 1
        if counter % 10000 == 0:
            print(f"Fetched {counter} images")
    print("Data fetched successfully")
    return np.array(X), np.array(y)

# function to create the model for classifying
def create_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# main execution
if __name__ == "__main__":
    X,y = fetch_data()

    # Split data into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split train+val into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
    X_val = X_val.reshape(-1, 28, 28, 1) / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
    
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)
    
    model = create_model(input_shape=(28,28,1), num_classes=10)
    print("Creating model...")
    model.fit(X_train,y_train, epochs=10, batch_size=64, verbose=2, validation_data=(X_val, y_val))
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test accuracy: {test_accuracy}")
    print(f"Test loss: {test_loss}")
