import pymongo
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from aux_functions import plot_label_distribution

input_shape = (28, 28, 1)
num_classes = 10

# function to fetch data from MongoDB
def fetch_data(client=None):
    if client is None:
        client = pymongo.MongoClient("mongodb://localhost:27017/")
    
    db = client.get_database('mnist')
    collection = db.get_collection('images')
    X = []
    y = []
    for doc in collection.find():
        X.append(doc['image'])
        y.append(doc['label'])
    return np.array(X), np.array(y)

# Fetch the data directly
X, y = fetch_data()

test_size = 0.2
val_size = 0.2
random_state = 42

# First, split off the test set
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Then split the remaining data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=random_state)

# reshape the data
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)

print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")
print(X_val.shape[0], "validation samples")

# normalize the data
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255
X_val = X_val.astype("float32") / 255

mean = np.mean(X_train)
std = np.std(X_train)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
X_val = (X_val - mean) / std

# plot the label distribution
plot_label_distribution(y_train, y_val, y_test)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

# we now build the model, a simple 2D Convolutional Neural Network
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

epochs = 15
batch_size = 64

# compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Define early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=2)

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.show()

# Evaluate the model on the test set
print("Evaluating the model on the test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")
print(f"Test loss: {test_loss}")