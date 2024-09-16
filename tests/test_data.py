import pytest
import numpy as np
from unittest.mock import Mock
from nn_split_manual import fetch_data, preprocess_data

def create_mock_client():
    mock_client = Mock()
    mock_db = Mock()
    mock_collection = Mock()
    
    # Simulate the MongoDB document structure
    mock_docs = [
        {'image': np.random.rand(28, 28), 'label': np.random.randint(0, 10)}
        for _ in range(100)  # Create 100 mock documents
    ]
    
    mock_collection.find.return_value = mock_docs
    mock_db.get_collection.return_value = mock_collection
    mock_client.get_database.return_value = mock_db
    
    return mock_client

def test_fetch_data():
    mock_client = create_mock_client()
    
    X, y = fetch_data(mock_client)
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == 100  # We created 100 mock documents
    assert X.shape[1:] == (28, 28)  # MNIST images are 28x28
    assert y.shape == (100,)  # 100 labels, one for each image
    assert np.all((y >= 0) & (y < 10))  # Labels should be between 0 and 9
    
    # Test that the client methods were called correctly
    mock_client.get_database.assert_called_once_with('mnist')
    mock_client.get_database().get_collection.assert_called_once_with('images')
    mock_client.get_database().get_collection().find.assert_called_once()