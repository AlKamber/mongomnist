# test_model.py
import pytest
from tensorflow import keras
from nn_split_manual import model, input_shape, num_classes

def test_model_architecture():
    # Test 1: Check if the model is a Sequential model
    assert isinstance(model, keras.Sequential)

    # Test 2: Check the number of layers
    assert len(model.layers) == 8

    # Test 3: Check input shape
    assert model.layers[0].input_shape == (None,) + input_shape

    # Test 4: Check first Conv2D layer
    assert isinstance(model.layers[0], keras.layers.Conv2D)
    assert model.layers[0].filters == 32
    assert model.layers[0].kernel_size == (3, 3)
    assert model.layers[0].activation.__name__ == 'relu'

    # Test 5: Check first MaxPooling2D layer
    assert isinstance(model.layers[1], keras.layers.MaxPooling2D)
    assert model.layers[1].pool_size == (2, 2)

    # Test 6: Check second Conv2D layer
    assert isinstance(model.layers[2], keras.layers.Conv2D)
    assert model.layers[2].filters == 64
    assert model.layers[2].kernel_size == (3, 3)
    assert model.layers[2].activation.__name__ == 'relu'

    # Test 7: Check second MaxPooling2D layer
    assert isinstance(model.layers[3], keras.layers.MaxPooling2D)
    assert model.layers[3].pool_size == (2, 2)

    # Test 8: Check Flatten layer
    assert isinstance(model.layers[4], keras.layers.Flatten)

    # Test 9: Check Dropout layer
    assert isinstance(model.layers[5], keras.layers.Dropout)
    assert model.layers[5].rate == 0.5

    # Test 10: Check output Dense layer
    assert isinstance(model.layers[6], keras.layers.Dense)
    assert model.layers[6].units == num_classes
    assert model.layers[6].activation.__name__ == 'softmax'

    # Test 11: Check model output shape
    assert model.output_shape == (None, num_classes)