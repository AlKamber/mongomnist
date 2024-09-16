import pytest
import numpy as np
from keras.models import Sequential
from main import create_model

def test_create_model():
    model = create_model()
    
    assert isinstance(model, Sequential)
    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 10)
    assert len(model.layers) == 8
    assert model.optimizer is not None
    assert model.loss is not None
    assert model.metrics is not None