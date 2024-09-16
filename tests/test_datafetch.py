import pytest
from unittest.mock import Mock, patch
import numpy as np
from main import fetch_data, get_mongo_client

@pytest.fixture
def mock_mongo_client():
    mock_client = Mock()
    mock_db = Mock()
    mock_collection = Mock()
    
    # Create mock data
    mock_data = [
        {'image': np.random.rand(28, 28).tolist(), 'label': np.random.randint(0, 10)}
        for _ in range(100)
    ]
    
    mock_collection.find.return_value = mock_data
    mock_db.images = mock_collection
    mock_client.mnist = mock_db
    
    return mock_client

@patch('main.get_mongo_client')
def test_fetch_data_without_client(mock_get_client, mock_mongo_client):
    mock_get_client.return_value = mock_mongo_client
    X, y = fetch_data()
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (100, 28, 28)
    assert y.shape == (100,)

def test_fetch_data_with_client(mock_mongo_client):
    X, y = fetch_data(mock_mongo_client)
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (100, 28, 28)
    assert y.shape == (100,)