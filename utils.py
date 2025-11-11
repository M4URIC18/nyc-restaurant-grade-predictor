"""
Utility functions for the Restaurant Inspection Grade Prediction app.
"""
import pickle
import os
import numpy as np
import pandas as pd
from pathlib import Path
import requests
from typing import Tuple, Optional


def load_model(model_path=None):
    """
    Load the trained RandomForestClassifier model.
    
    Args:
        model_path: Path to the model file. If None, tries common locations.
        
    Returns:
        Loaded model object
        
    Raises:
        FileNotFoundError: If model file is not found
        Exception: If model cannot be loaded
    """
    if model_path is None:
        # Try multiple possible locations
        possible_paths = [
            'app/baseline_model.pkl',
            'app/model.pkl',
            'baseline_model.pkl',
            'model.pkl',
            os.path.join(os.path.dirname(__file__), 'app', 'baseline_model.pkl'),
            os.path.join(os.path.dirname(__file__), 'baseline_model.pkl'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(
                "Model file not found. Please ensure 'baseline_model.pkl' or 'model.pkl' "
                "exists in the current directory or 'app' folder."
            )
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")


def get_feature_names():
    """
    Get the expected feature names for the model.
    
    Returns:
        List of feature names
    """
    return [
        'borough',
        'cuisine_description',
        'inspection_type',
        'nfh_indexscore',
        'median_income',
        'poverty_rate'
    ]


def prepare_input_data(borough, cuisine_description, inspection_type, 
                      nfh_indexscore, median_income, poverty_rate):
    """
    Prepare input data for prediction.
    
    Args:
        borough: Borough name (str)
        cuisine_description: Cuisine type (str)
        inspection_type: Inspection type (str)
        nfh_indexscore: NFH index score (float)
        median_income: Median income (float)
        poverty_rate: Poverty rate (float)
        
    Returns:
        DataFrame with prepared features
    """
    data = {
        'borough': [borough],
        'cuisine_description': [cuisine_description],
        'inspection_type': [inspection_type],
        'nfh_indexscore': [float(nfh_indexscore)],
        'median_income': [float(median_income)],
        'poverty_rate': [float(poverty_rate)]
    }
    
    return pd.DataFrame(data)


def predict_grade(model, input_data):
    """
    Predict restaurant inspection grade.
    
    Args:
        model: Trained model (can be a Pipeline or standalone model)
        input_data: DataFrame with input features
        
    Returns:
        Predicted grade (str) and prediction probabilities (dict)
    """
    try:
        # Handle both Pipeline and standalone models
        # If it's a Pipeline, it will handle preprocessing
        # If it's a standalone model, it expects preprocessed data
        
        # Get prediction
        prediction = model.predict(input_data)[0]
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(input_data)[0]
                classes = model.classes_
                prob_dict = {class_name: float(prob) for class_name, prob in zip(classes, probabilities)}
            except Exception as prob_error:
                # If predict_proba fails, just return empty dict
                prob_dict = {}
        else:
            prob_dict = {}
        
        return prediction, prob_dict
    except ValueError as e:
        # Common error: feature mismatch
        error_msg = str(e)
        if "feature" in error_msg.lower() or "column" in error_msg.lower():
            raise Exception(
                f"Feature mismatch error: {error_msg}\n\n"
                "This might indicate that the model was trained with different features or encoding. "
                "Please ensure the model expects the same feature format as provided."
            )
        raise Exception(f"Error making prediction: {error_msg}")
    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")


def get_feature_importances(model, feature_names=None):
    """
    Get feature importances from the model.
    
    Args:
        model: Trained model (can be a Pipeline or standalone model)
        feature_names: List of feature names (optional)
        
    Returns:
        Dictionary of feature names and their importances
    """
    # Handle Pipeline models - extract the actual model
    actual_model = model
    if hasattr(model, 'named_steps'):
        # It's a Pipeline, try to find the classifier step
        for step_name, step in model.named_steps.items():
            if hasattr(step, 'feature_importances_') or hasattr(step, 'predict'):
                actual_model = step
                break
    
    if not hasattr(actual_model, 'feature_importances_'):
        return {}
    
    if feature_names is None:
        feature_names = get_feature_names()
    
    importances = actual_model.feature_importances_
    
    # If model is a Pipeline, feature names might be different
    # Try to get feature names from the model if available
    if hasattr(actual_model, 'feature_names_in_'):
        # Use the feature names the model was trained with
        model_feature_names = actual_model.feature_names_in_
        if len(model_feature_names) == len(importances):
            feature_names = model_feature_names
    
    # Ensure we have the right number of feature names
    if len(feature_names) != len(importances):
        # Use generic names if mismatch
        feature_names = [f"Feature_{i+1}" for i in range(len(importances))]
    
    # Create dictionary and sort by importance
    importance_dict = {
        feature: float(importance) 
        for feature, importance in zip(feature_names, importances)
    }
    
    # Sort by importance (descending)
    sorted_importances = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )
    
    return sorted_importances


def validate_inputs(borough, cuisine_description, inspection_type,
                   nfh_indexscore, median_income, poverty_rate):
    """
    Validate user inputs.
    
    Args:
        All input parameters
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for empty strings
    if not borough or not cuisine_description or not inspection_type:
        return False, "Please fill in all categorical fields (borough, cuisine, inspection type)."
    
    # Validate numerical inputs
    try:
        nfh_indexscore = float(nfh_indexscore)
        median_income = float(median_income)
        poverty_rate = float(poverty_rate)
    except (ValueError, TypeError):
        return False, "Please enter valid numbers for numerical fields."
    
    # Validate ranges (reasonable checks)
    if poverty_rate < 0 or poverty_rate > 100:
        return False, "Poverty rate must be between 0 and 100."
    
    if median_income < 0:
        return False, "Median income must be positive."
    
    return True, ""


def geocode_address(address: str, api_key: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Geocode an address to get latitude and longitude using Google Maps Geocoding API.
    
    Args:
        address: Address string to geocode
        api_key: Google Maps API key (optional, can be set via GOOGLE_MAPS_API_KEY env var)
        
    Returns:
        Tuple of (latitude, longitude, formatted_address) or (None, None, None) if failed
    """
    if not api_key:
        api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    
    if not api_key:
        return None, None, None
    
    if not address or not address.strip():
        return None, None, None
    
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': address,
            'key': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == 'OK' and data['results']:
            location = data['results'][0]['geometry']['location']
            lat = location['lat']
            lng = location['lng']
            formatted_address = data['results'][0]['formatted_address']
            return lat, lng, formatted_address
        else:
            return None, None, None
    except Exception as e:
        # Silent fail - return None values
        return None, None, None

