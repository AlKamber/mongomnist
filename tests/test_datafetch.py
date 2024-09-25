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
        assert X.shape[0] > 0
        assert y.shape[0] > 0
        print("MongoDB client test successful. Sample documents:", X)
    except Exception as e:
        pytest.fail(f"MongoDB client test failed: {e}")
        
test_real_mongo_client()