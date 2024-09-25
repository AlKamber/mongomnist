import os
import pytest
from unittest.mock import Mock, patch
import numpy as np
from main import fetch_data, get_mongo_client
from dotenv import load_dotenv
from urllib.parse import quote_plus

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
