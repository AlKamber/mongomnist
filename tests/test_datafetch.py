import os
import sys
import pytest
from unittest.mock import Mock, patch
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import fetch_data, get_mongo_client

def test_real_mongo_client():
    # Access the environment variables
    mongo_username = os.getenv('MONGO_USERNAME')
    mongo_password = os.getenv('MONGO_PASSWORD')

    assert mongo_username is not None, "MONGO_USERNAME not found"
    assert mongo_password is not None, "MONGO_PASSWORD not found"
    
    # Get the real MongoDB client
    client = get_mongo_client()
    
    try:
        X, y = fetch_data(client)
        unique_labels = np.unique(y)
        
        assert len(X) == 70000, "X should have 70000 samples"
        assert len(y) == 70000, "y should have 70000 samples"
        assert len(unique_labels) == 10, "Expected 10 unique classes (digits 0-9)."
        assert np.all((y >= 0) & (y <= 9)), "Labels should be in the range 0-9."
        assert X.dtype == np.uint8, "Image data type should be uint8."
        assert np.all((X >= 0) & (X <= 255)), "Pixel values should be in the range 0-255."
        assert not np.any(np.isnan(X)), "There should be no missing values in the images."
        assert not np.any(np.isnan(y)), "There should be no missing values in the labels."
        
        print("MongoDB client test successful.")
    except Exception as e:
        pytest.fail(f"MongoDB client test failed: {e}")
        
test_real_mongo_client()