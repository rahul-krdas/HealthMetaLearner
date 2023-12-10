# health_meta_learner/utils.py
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """
    Preprocess the input data using standard scaling.

    Parameters:
    - data: Input data to be preprocessed.

    Returns:
    - Preprocessed data.
    """
    scaler = StandardScaler()
    preprocessed_data = scaler.fit_transform(data)
    return preprocessed_data

def helper_function():
    """
    A placeholder for a helper function that might be needed in the package.

    Parameters: None

    Returns: None
    """
    print("This is a placeholder for a helpful utility function.")
