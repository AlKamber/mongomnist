import pytest
import numpy as np
from keras.models import Sequential
from main import create_model

def test_create_model():
    input_shape = (28, 28, 1)
    num_classes = 10
    model = create_model(input_shape, num_classes)
    
    assert isinstance(model, Sequential)
    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 10)
    assert len(model.layers) == 8
    assert model.optimizer is not None
    assert model.loss is not None
    assert model.metrics is not None

def test_model_predict():
    input_shape = (28, 28, 1)
    num_classes = 10
    model = create_model(input_shape, num_classes)
    
    dummy_input = np.random.rand(1, 28, 28, 1)
    prediction = model.predict(dummy_input)
    
    assert prediction.shape == (1, 10)
    assert np.isclose(np.sum(prediction), 1.0)
    assert np.all(prediction >= 0) and np.all(prediction <= 1)