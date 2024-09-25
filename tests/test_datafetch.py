import os
import sys
import pytest
from unittest.mock import Mock, patch
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import fetch_data, get_mongo_client

def test_real_mongo_client():
    # Get the real MongoDB client
    client = get_mongo_client()
    
    try:
        X, y = fetch_data(client)
        assert X.shape[0] > 0
        assert y.shape[0] > 0
        print("MongoDB client test successful. Sample documents:", X)
    except Exception as e:
        pytest.fail(f"MongoDB client test failed: {e}")