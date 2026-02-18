"""
Heart Disease Detection - Preprocessing Module
===============================================
Handles data preprocessing for prediction.

Author: Heart Disease Detection Project
Date: 2024
"""

import numpy as np
from typing import Dict, Any, List


def preprocess_input(
    input_data: Dict[str, Any], 
    feature_names: List[str]
) -> np.ndarray:
    """
    Preprocess input data for model prediction.
    
    Args:
        input_data: Dictionary containing patient features
        feature_names: List of feature names in correct order
        
    Returns:
        Numpy array of preprocessed features
    """
    # Validate all required features are present
    validate_input(input_data, feature_names)
    
    # Extract features in correct order
    features = []
    for feature in feature_names:
        value = input_data.get(feature)
        features.append(float(value))
    
    # Convert to numpy array and reshape for single prediction
    feature_array = np.array(features).reshape(1, -1)
    
    return feature_array


def validate_input(input_data: Dict[str, Any], feature_names: List[str]) -> None:
    """
    Validate input data for required features and value ranges.
    
    Args:
        input_data: Dictionary containing patient features
        feature_names: List of required feature names
        
    Raises:
        ValueError: If validation fails
    """
    # Define validation rules for each feature
    validation_rules = {
        'age': {'min': 1, 'max': 120, 'type': int},
        'sex': {'min': 0, 'max': 1, 'type': int},
        'chest_pain_type': {'min': 0, 'max': 3, 'type': int},
        'resting_blood_pressure': {'min': 50, 'max': 300, 'type': int},
        'cholesterol': {'min': 100, 'max': 600, 'type': int},
        'fasting_blood_sugar': {'min': 0, 'max': 1, 'type': int},
        'resting_ecg': {'min': 0, 'max': 2, 'type': int},
        'max_heart_rate': {'min': 60, 'max': 250, 'type': int},
        'exercise_induced_angina': {'min': 0, 'max': 1, 'type': int},
        'st_depression': {'min': 0.0, 'max': 10.0, 'type': float},
        'st_slope': {'min': 0, 'max': 2, 'type': int},
        'num_major_vessels': {'min': 0, 'max': 4, 'type': int},
        'thalassemia': {'min': 0, 'max': 3, 'type': int}
    }
    
    # Check for missing features
    missing_features = []
    for feature in feature_names:
        if feature not in input_data or input_data[feature] is None:
            missing_features.append(feature)
    
    if missing_features:
        raise ValueError(f"Missing required features: {', '.join(missing_features)}")
    
    # Validate each feature
    for feature in feature_names:
        value = input_data[feature]
        
        if feature in validation_rules:
            rules = validation_rules[feature]
            
            # Check type
            try:
                value = rules['type'](value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid type for '{feature}': expected {rules['type'].__name__}"
                )
            
            # Check range
            if value < rules['min'] or value > rules['max']:
                raise ValueError(
                    f"Value for '{feature}' ({value}) out of range: "
                    f"expected {rules['min']} to {rules['max']}"
                )


def get_feature_descriptions() -> Dict[str, Dict[str, Any]]:
    """
    Get detailed descriptions of all features.
    
    Returns:
        Dictionary with feature descriptions
    """
    return {
        'age': {
            'description': 'Age of the patient in years',
            'type': 'int',
            'range': (1, 120),
            'unit': 'years'
        },
        'sex': {
            'description': 'Gender of the patient',
            'type': 'int',
            'values': {0: 'Female', 1: 'Male'}
        },
        'chest_pain_type': {
            'description': 'Type of chest pain experienced',
            'type': 'int',
            'values': {
                0: 'Typical Angina',
                1: 'Atypical Angina',
                2: 'Non-anginal Pain',
                3: 'Asymptomatic'
            }
        },
        'resting_blood_pressure': {
            'description': 'Resting blood pressure on admission',
            'type': 'int',
            'range': (50, 300),
            'unit': 'mm Hg'
        },
        'cholesterol': {
            'description': 'Serum cholesterol level',
            'type': 'int',
            'range': (100, 600),
            'unit': 'mg/dl'
        },
        'fasting_blood_sugar': {
            'description': 'Fasting blood sugar > 120 mg/dl',
            'type': 'int',
            'values': {0: 'False (≤120 mg/dl)', 1: 'True (>120 mg/dl)'}
        },
        'resting_ecg': {
            'description': 'Resting electrocardiographic results',
            'type': 'int',
            'values': {
                0: 'Normal',
                1: 'ST-T wave abnormality',
                2: 'Left ventricular hypertrophy'
            }
        },
        'max_heart_rate': {
            'description': 'Maximum heart rate achieved during exercise',
            'type': 'int',
            'range': (60, 250),
            'unit': 'bpm'
        },
        'exercise_induced_angina': {
            'description': 'Angina induced by exercise',
            'type': 'int',
            'values': {0: 'No', 1: 'Yes'}
        },
        'st_depression': {
            'description': 'ST depression induced by exercise relative to rest',
            'type': 'float',
            'range': (0.0, 10.0)
        },
        'st_slope': {
            'description': 'Slope of the peak exercise ST segment',
            'type': 'int',
            'values': {
                0: 'Upsloping',
                1: 'Flat',
                2: 'Downsloping'
            }
        },
        'num_major_vessels': {
            'description': 'Number of major vessels colored by fluoroscopy',
            'type': 'int',
            'range': (0, 4)
        },
        'thalassemia': {
            'description': 'Thalassemia blood disorder type',
            'type': 'int',
            'values': {
                0: 'Normal',
                1: 'Fixed Defect',
                2: 'Reversible Defect',
                3: 'Not Described'
            }
        }
    }


def encode_categorical_features(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Encode categorical string values to integers.
    
    This function handles cases where input data may contain
    string labels instead of numeric values.
    
    Args:
        data: Input data dictionary
        
    Returns:
        Encoded data dictionary
    """
    # Encoding mappings for categorical string values
    encodings = {
        'sex': {'female': 0, 'male': 1, 'f': 0, 'm': 1},
        'chest_pain_type': {
            'typical angina': 0, 
            'atypical angina': 1,
            'non-anginal pain': 2, 
            'asymptomatic': 3
        },
        'fasting_blood_sugar': {
            'false': 0, 'true': 1, 
            'no': 0, 'yes': 1,
            '<=120': 0, '>120': 1
        },
        'resting_ecg': {
            'normal': 0,
            'st-t wave abnormality': 1,
            'left ventricular hypertrophy': 2,
            'lvh': 2
        },
        'exercise_induced_angina': {
            'no': 0, 'yes': 1,
            'false': 0, 'true': 1
        },
        'st_slope': {
            'upsloping': 0, 'flat': 1, 'downsloping': 2,
            'up': 0, 'down': 2
        },
        'thalassemia': {
            'normal': 0,
            'fixed defect': 1,
            'reversible defect': 2,
            'not described': 3
        }
    }
    
    encoded_data = data.copy()
    
    for feature, mapping in encodings.items():
        if feature in encoded_data:
            value = encoded_data[feature]
            if isinstance(value, str):
                value_lower = value.lower().strip()
                if value_lower in mapping:
                    encoded_data[feature] = mapping[value_lower]
    
    return encoded_data


# ============================================
# TEST PREPROCESSING (FOR DEVELOPMENT)
# ============================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("TESTING PREPROCESSING MODULE")
    print("=" * 50)
    
    # Test feature names
    feature_names = [
        'age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
        'cholesterol', 'fasting_blood_sugar', 'resting_ecg',
        'max_heart_rate', 'exercise_induced_angina', 'st_depression',
        'st_slope', 'num_major_vessels', 'thalassemia'
    ]
    
    # Sample input data
    sample_input = {
        'age': 55,
        'sex': 1,
        'chest_pain_type': 2,
        'resting_blood_pressure': 130,
        'cholesterol': 250,
        'fasting_blood_sugar': 0,
        'resting_ecg': 1,
        'max_heart_rate': 150,
        'exercise_induced_angina': 0,
        'st_depression': 1.5,
        'st_slope': 1,
        'num_major_vessels': 1,
        'thalassemia': 2
    }
    
    print("\n📋 Input Data:")
    for key, value in sample_input.items():
        print(f"   • {key}: {value}")
    
    # Test preprocessing
    try:
        processed = preprocess_input(sample_input, feature_names)
        print("\n✅ Preprocessing successful!")
        print(f"   • Output shape: {processed.shape}")
        print(f"   • Output: {processed}")
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
    
    # Test validation with invalid data
    print("\n🔍 Testing validation with invalid data...")
    invalid_input = sample_input.copy()
    invalid_input['age'] = 150  # Invalid age
    
    try:
        preprocess_input(invalid_input, feature_names)
        print("   ❌ Validation should have failed!")
    except ValueError as e:
        print(f"   ✅ Validation correctly caught error: {e}")
