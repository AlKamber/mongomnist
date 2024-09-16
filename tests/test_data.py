import pytest
import numpy as np
from nn_split_manual import preprocess_data
    
def test_preprocess_data():
    # Create a sample input
    X = np.random.randint(0, 256, size=(100, 28, 28)).astype(np.uint8)
    
    # Preprocess the data
    X_processed = preprocess_data(X)
    
    # Test 1: Check if the output is float32
    assert X_processed.dtype == np.float32  # Processed data should be float32
    
    # Test 2: Check if the mean is close to 0
    assert np.isclose(np.mean(X_processed), 0, atol=1e-6)  # Processed data should have mean close to 0
    
    X_processed = preprocess_data(X)
    
    # Test 3: Check if the standard deviation is close to 1
    assert np.isclose(np.std(X_processed), 1, atol=1e-6)  # Processed data should have std close to 1
    
    # Test 4: Check if the shape is correct
    assert X_processed.shape == (100, 28, 28, 1)
    
    # Test 5: Check if the function handles zero variance input
    X_constant = np.full((10, 28, 28), 128, dtype=np.uint8)
    X_constant_processed = preprocess_data(X_constant)
    assert not np.any(np.isnan(X_constant_processed))  # Function should handle constant input without producing NaNs

