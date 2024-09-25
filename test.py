import numpy as np
from keras.datasets import mnist  # or any other method you use to load MNIST

def load_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Combine train and test sets if needed
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    return X, y

def check_pixel_data_type():
    X, y = load_mnist_data()
    print("Data type of pixel values:", X.dtype)  # Check the data type of pixel values

if __name__ == "__main__":
    check_pixel_data_type()