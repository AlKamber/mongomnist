import pymongo
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# function to get the mongo client
def get_mongo_client():
    return pymongo.MongoClient("mongodb://localhost:27017/")

# function to get the data from mongoDB
def fetch_data(client=None):
    if client is None:
        client = get_mongo_client()
    db = client.mnist
    collection = db.images
    X = []
    y = []
    for doc in collection.find():
        X.append(doc['image'])
        y.append(doc['label'])
    return np.array(X), np.array(y)

# function to create the model for classifying
def create_model(input_shape, num_classes):
    model = Sequential([
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
    model.fit(X_train,y_train, epochs=10, batch_size=32, verbose=2, validation_data=(X_val, y_val))
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy}")
    print(f"Test loss: {test_loss}")
