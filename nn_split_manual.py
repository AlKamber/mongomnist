import pymongo
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

input_shape = (28, 28, 1)
num_classes = 10

# Connect to MongoDB on localhost:27017
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["mnist_database_no_split"]
collection = db["images"]

def plot_label_distribution(y_train, y_val, y_test):
    labels = range(10)  # 0 to 9
    train_counts = [np.sum(y_train == i) for i in labels]
    val_counts = [np.sum(y_val == i) for i in labels]
    test_counts = [np.sum(y_test == i) for i in labels]
    
    train_percentages = [count / len(y_train) * 100 for count in train_counts]
    val_percentages = [count / len(y_val) * 100 for count in val_counts]
    test_percentages = [count / len(y_test) * 100 for count in test_counts]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, train_percentages, width, label='Train')
    rects2 = ax.bar(x, val_percentages, width, label='Validation')
    rects3 = ax.bar(x + width, test_percentages, width, label='Test')

    ax.set_ylabel('Percentage')
    ax.set_xlabel('Digit')
    ax.set_title('Label Distribution in Train, Validation, and Test Sets')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.show()

# define a function to fetch data from MongoDB
def fetch_data():
    data = []
    labels = []
    for doc in collection.find():
        data.append(np.array(doc["image"]))
        labels.append(doc["label"])
    print("Finished loading {} elements".format(len(data)))
    return np.array(data), np.array(labels)

# fetch the data with the function we defined above
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
# This step performs two types of normalization:

# 1. Min-Max Scaling:
#    X_train and X_test were divided by 255 to scale pixel values to [0, 1] range.
#    This is because pixel values originally range from 0 to 255.

# 2. Z-score Normalization (Standardization):
#    We calculate the mean and standard deviation of the training data.
#    Then, we subtract the mean from each sample and divide by the standard deviation.
#    This centers the data around 0 and gives it unit variance.
#    We apply the same transformation to both training and test data using
#    the statistics (mean and std) computed from the training data only.

# Benefits of normalization:
# - Helps in faster convergence during training
# - Prevents certain features from dominating due to their scale
# - Often leads to better model performance and generalization
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255
X_val = X_val.astype("float32") / 255

mean = np.mean(X_train)
std = np.std(X_train)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
X_val = (X_val - mean) / std

plot_label_distribution(y_train, y_val, y_test)

# convert the labels to categorical
# The labels are converted to categorical (one-hot encoded) for the following reasons:
# 1. Multi-class classification: MNIST has 10 classes (digits 0-9), and we're using a softmax activation
#    in the output layer. Categorical labels work well with softmax for multi-class problems.
# 2. Loss function compatibility: We're using 'categorical_crossentropy' as our loss function, which
#    expects the labels to be in categorical format.
# 3. Model output format: The model's output will be a probability distribution over the 10 classes.
#    Having the labels in the same format allows for direct comparison and easier interpretation.
# 4. Avoiding ordinal relationships: One-hot encoding ensures that the model doesn't assume any
#    ordinal relationship between the classes (e.g., that 9 is "greater than" 1 in terms of classification).

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