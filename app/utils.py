"""
Utility functions for the Restaurant Inspection Grade Prediction app.
"""
import pickle
import os
import numpy as np
import pandas as pd
from pathlib import Path
import requests
from typing import Tuple, Optional, Dict


def load_model(model_path=None):
    """
    Load the trained model saved with joblib.
    Works both locally and on Streamlit Cloud.
    """
    import os
    import joblib

    # Directories to search
    app_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(app_dir)
    root_dir = os.path.dirname(parent_dir)

    # Possible model paths
    possible_paths = [
        model_path,
        os.path.join(app_dir, 'baseline_model.pkl'),
        os.path.join(parent_dir, 'baseline_model.pkl'),
        os.path.join(root_dir, 'baseline_model.pkl'),
        os.path.join(app_dir, 'model.pkl'),
        os.path.join(parent_dir, 'model.pkl'),
        os.path.join(root_dir, 'model.pkl'),
    ]
    possible_paths = [p for p in possible_paths if p]

    # Try to load model
    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                return model
            except Exception as e:
                raise Exception(f"❌ Failed to load model from {path}: {e}")

    raise FileNotFoundError(
        "❌ Model file not found.\n"
        "Make sure 'baseline_model.pkl' or 'model.pkl' exists in the root or app/ folder.\n"
        "Expected in one of:\n" + "\n".join(possible_paths)
    )




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
    Prepare a single input row matching the model's expected features.
    The model expects: 
    ['boro', 'cuisine_description', 'score', 'median_income', 'poverty_rate',
     'perc_white', 'perc_black', 'perc_asian', 'perc_hispanic', 'indexscore']
    """
    # Rename borough → boro
    boro = borough

    # Create a base dataframe with the correct structure
    input_dict = {
        'boro': [boro],
        'cuisine_description': [cuisine_description],
        'score': [10.0],               # Placeholder (e.g., average inspection score)
        'median_income': [median_income],
        'poverty_rate': [poverty_rate],
        'perc_white': [0.4],           # Default demographic placeholders
        'perc_black': [0.2],
        'perc_asian': [0.2],
        'perc_hispanic': [0.2],
        'indexscore': [nfh_indexscore] # Map app’s NFH score to model’s indexscore
    }

    df_input = pd.DataFrame(input_dict)

    return df_input



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
    Supports both Pipeline models (with named_steps) and standalone models.
    
    Args:
        model: Trained model (can be a Pipeline or standalone model)
        feature_names: List of feature names (optional)
        
    Returns:
        Dictionary of feature names and their importances
    """
    # Handle Pipeline models - try to get RandomForestClassifier from named_steps
    actual_model = model
    if hasattr(model, 'named_steps'):
        # It's a Pipeline, try to find the RandomForestClassifier step
        # Try common step names
        possible_step_names = ['randomforestclassifier', 'classifier', 'model', 'rf', 'random_forest']
        
        for step_name in possible_step_names:
            if step_name.lower() in [name.lower() for name in model.named_steps.keys()]:
                # Find the exact key (case-insensitive)
                for key in model.named_steps.keys():
                    if key.lower() == step_name.lower():
                        actual_model = model.named_steps[key]
                        break
                if hasattr(actual_model, 'feature_importances_'):
                    break
        
        # If not found by name, try to find any step with feature_importances_
        if not hasattr(actual_model, 'feature_importances_'):
            for step_name, step in model.named_steps.items():
                if hasattr(step, 'feature_importances_'):
                    actual_model = step
                    break
    
    if not hasattr(actual_model, 'feature_importances_'):
        return {}
    
    if feature_names is None:
        feature_names = get_feature_names()
    
    importances = actual_model.feature_importances_
    
    # If model is a Pipeline, feature names might be different after encoding
    # Try to get feature names from the model if available
    if hasattr(actual_model, 'feature_names_in_'):
        # Use the feature names the model was trained with
        model_feature_names = actual_model.feature_names_in_
        if len(model_feature_names) == len(importances):
            feature_names = list(model_feature_names)
    
    # If we still have a mismatch, try to map back to original feature names
    # This handles the case where one-hot encoding created more features
    if len(feature_names) != len(importances):
        # If we have more importances than original features, it's likely one-hot encoded
        # Return the importances with generic names or try to infer
        if len(importances) > len(get_feature_names()):
            # Likely one-hot encoded - use generic names
            feature_names = [f"Feature_{i+1}" for i in range(len(importances))]
        else:
            # Use original feature names, truncating if necessary
            feature_names = get_feature_names()[:len(importances)]
    
    # Create dictionary and sort by importance
    importance_dict = {
        str(feature): float(importance) 
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
    if not borough or not str(cuisine_description).strip() or not inspection_type:
        return False, "Please fill in all categorical fields (borough, cuisine description, inspection type)."
    
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
    
    if nfh_indexscore < 0:
        return False, "NFH index score must be non-negative."
    
    return True, ""


def geocode_address(address: str, api_key: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[Dict]]:
    """
    Geocode an address to get latitude, longitude, formatted address, and address components.
    
    Args:
        address: Address string to geocode
        api_key: Google Maps API key (optional, can be set via GOOGLE_MAPS_API_KEY env var)
        
    Returns:
        Tuple of (latitude, longitude, formatted_address, address_components) or (None, None, None, None) if failed
    """
    if not api_key:
        api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    
    if not api_key:
        return None, None, None, None
    
    if not address or not address.strip():
        return None, None, None, None
    
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
            result = data['results'][0]
            location = result['geometry']['location']
            lat = location['lat']
            lng = location['lng']
            formatted_address = result['formatted_address']
            
            # Extract address components
            address_components = {}
            for component in result.get('address_components', []):
                types = component.get('types', [])
                if 'administrative_area_level_2' in types:  # County (Borough in NYC)
                    address_components['borough'] = component.get('long_name', '')
                elif 'locality' in types:
                    address_components['city'] = component.get('long_name', '')
                elif 'administrative_area_level_1' in types:
                    address_components['state'] = component.get('short_name', '')
                elif 'postal_code' in types:
                    address_components['postal_code'] = component.get('long_name', '')
            
            return lat, lng, formatted_address, address_components
        else:
            return None, None, None, None
    except Exception as e:
        # Silent fail - return None values
        return None, None, None, None


def reverse_geocode(lat: float, lng: float, api_key: Optional[str] = None) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Reverse geocode coordinates to get address and address components.
    
    Args:
        lat: Latitude
        lng: Longitude
        api_key: Google Maps API key (optional, can be set via GOOGLE_MAPS_API_KEY env var)
        
    Returns:
        Tuple of (formatted_address, address_components) or (None, None) if failed
    """
    if not api_key:
        api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    
    if not api_key:
        return None, None
    
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'latlng': f"{lat},{lng}",
            'key': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == 'OK' and data['results']:
            result = data['results'][0]
            formatted_address = result['formatted_address']
            
            # Extract address components
            address_components = {}
            for component in result.get('address_components', []):
                types = component.get('types', [])
                if 'administrative_area_level_2' in types:  # County (Borough in NYC)
                    address_components['borough'] = component.get('long_name', '')
                elif 'locality' in types:
                    address_components['city'] = component.get('long_name', '')
                elif 'administrative_area_level_1' in types:
                    address_components['state'] = component.get('short_name', '')
                elif 'postal_code' in types:
                    address_components['postal_code'] = component.get('long_name', '')
                elif 'sublocality' in types or 'sublocality_level_1' in types:
                    # Sometimes borough is in sublocality for NYC
                    if 'borough' not in address_components:
                        address_components['borough'] = component.get('long_name', '')
            
            return formatted_address, address_components
        else:
            return None, None
    except Exception as e:
        # Silent fail - return None values
        return None, None


def normalize_borough_name(borough: str) -> str:
    """
    Normalize borough name from geocoding to match our dropdown options.
    
    Args:
        borough: Borough name from geocoding
        
    Returns:
        Normalized borough name matching our options
    """
    borough_lower = borough.lower().strip()
    
    # Map common variations to our standard names
    borough_mapping = {
        'manhattan': 'Manhattan',
        'new york': 'Manhattan',
        'nyc': 'Manhattan',
        'brooklyn': 'Brooklyn',
        'kings': 'Brooklyn',
        'queens': 'Queens',
        'bronx': 'Bronx',
        'the bronx': 'Bronx',
        'staten island': 'Staten Island',
        'richmond': 'Staten Island'
    }
    
    # Check for partial matches
    for key, value in borough_mapping.items():
        if key in borough_lower:
            return value
    
    # If no match, return as is (capitalized)
    return borough.strip().title()

