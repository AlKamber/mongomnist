#%%

import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from dotenv import load_dotenv
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
import matplotlib.pyplot as plt

# Suppress specific Keras warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
# Try to load .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

def get_mongo_client():
    username = quote_plus(os.getenv('MONGO_USERNAME'))
    password = quote_plus(os.getenv('MONGO_PASSWORD'))
    print(username, password)
    
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

def display_sample_images(X, y, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples*3, 3))
    for i in range(num_samples):
        idx = np.random.randint(0, len(X))
        axes[i].imshow(X[idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(f"Label: {y[idx]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    
def print_data_info(X, y, dataset_name):
    print(f"{dataset_name} set shape: {X.shape}")
    print(f"{dataset_name} set label shape: {y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Sample labels: {y[:10]}")
    print(f"Min pixel value: {X.min()}, Max pixel value: {X.max()}")
    
def preprocess_data(X, y):
    """Preprocess the data."""
    X = X.reshape(-1, 28, 28, 1) / 255.0
    y = to_categorical(y)
    return X, y

def plot_label_distribution(y_train, y_val, y_test):
    """
    Plot the distribution of labels in training, validation, and test sets.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    def plot_distribution(y, ax, title):
        unique, counts = np.unique(y, return_counts=True)
        ax.bar(unique, counts)
        ax.set_title(title)
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        ax.set_xticks(unique)
    
    plot_distribution(y_train, ax1, 'Training Set')
    plot_distribution(y_val, ax2, 'Validation Set')
    plot_distribution(y_test, ax3, 'Test Set')
    
    plt.tight_layout()
    plt.show()

#%%
# main execution
if __name__ == "__main__":
    
    X,y = fetch_data()
    
    #%%
    display_sample_images(X, y)

    # Split data into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split train+val into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    #%%
    plot_label_distribution(y_train, y_val, y_test)
    
    #%%
    X_train, y_train = preprocess_data(X_train, y_train)
    X_val, y_val = preprocess_data(X_val, y_val)
    X_test, y_test = preprocess_data(X_test, y_test)
    
    #%%
    model = create_model(input_shape=(28,28,1), num_classes=10)
    print("Creating model...")
    model.fit(X_train,y_train, epochs=10, batch_size=64, verbose=2, validation_data=(X_val, y_val))
    
    #%%
    # Save the model
    model.save("mnist_model.keras")
    
    #%%
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test accuracy: {test_accuracy}")
    print(f"Test loss: {test_loss}")
