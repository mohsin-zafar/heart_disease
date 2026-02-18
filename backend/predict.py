"""
Heart Disease Detection - Prediction Logic
===========================================
Handles model loading and prediction logic.

Author: Heart Disease Detection Project
Date: 2024
"""

import numpy as np
import joblib
import os
from typing import Dict, Any, List
from preprocess import preprocess_input


class HeartDiseasePredictor:
    """
    Heart Disease Prediction class.
    Handles model loading and prediction logic.
    """
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Initialize the predictor with model and scaler.
        
        Args:
            model_path: Path to the trained model file
            scaler_path: Path to the scaler file
        """
        # Default paths
        if model_path is None:
            model_path = self._get_model_path('model.pkl')
        if scaler_path is None:
            scaler_path = self._get_model_path('scaler.pkl')
        
        # Load model and scaler
        self.model = self._load_model(model_path)
        self.scaler = self._load_scaler(scaler_path)
        self.model_info = self._load_model_info()
        self.feature_names = self._load_feature_names()
        
        print("✅ Model and scaler loaded successfully!")
    
    def _get_model_path(self, filename: str) -> str:
        """
        Get the correct path to model files.
        Handles different directory structures.
        """
        # Try different possible paths
        possible_paths = [
            os.path.join('model', filename),
            os.path.join('..', 'model', filename),
            os.path.join(os.path.dirname(__file__), 'model', filename),
            os.path.join(os.path.dirname(__file__), '..', 'model', filename),
            filename
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Return default path if none exist
        return os.path.join('model', filename)
    
    def _load_model(self, model_path: str):
        """
        Load the trained model from disk.
        """
        try:
            model = joblib.load(model_path)
            print(f"📁 Model loaded from: {model_path}")
            return model
        except FileNotFoundError:
            print(f"⚠️ Warning: Model file not found at {model_path}")
            print("   Please ensure model.pkl is in the 'model' directory")
            return None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def _load_scaler(self, scaler_path: str):
        """
        Load the feature scaler from disk.
        """
        try:
            scaler = joblib.load(scaler_path)
            print(f"📁 Scaler loaded from: {scaler_path}")
            return scaler
        except FileNotFoundError:
            print(f"⚠️ Warning: Scaler file not found at {scaler_path}")
            print("   Predictions will be made without scaling (may affect accuracy)")
            return None
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
            return None
    
    def _load_model_info(self) -> Dict:
        """
        Load model metadata.
        """
        try:
            info_path = self._get_model_path('model_info.pkl')
            return joblib.load(info_path)
        except Exception:
            return {
                'model_name': 'Heart Disease Classifier',
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.0,
                'features': []
            }
    
    def _load_feature_names(self) -> List[str]:
        """
        Load feature names.
        """
        try:
            names_path = self._get_model_path('feature_names.pkl')
            return joblib.load(names_path)
        except Exception:
            return [
                'age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
                'cholesterol', 'fasting_blood_sugar', 'resting_ecg',
                'max_heart_rate', 'exercise_induced_angina', 'st_depression',
                'st_slope', 'num_major_vessels', 'thalassemia'
            ]
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction based on input patient data.
        
        Args:
            input_data: Dictionary containing patient features
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Cannot make predictions.")
        
        # Preprocess input data
        processed_data = preprocess_input(input_data, self.feature_names)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            scaled_data = self.scaler.transform(processed_data)
        else:
            scaled_data = processed_data
        
        # Make prediction
        prediction = self.model.predict(scaled_data)[0]
        
        # Get probability (if model supports it)
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(scaled_data)[0][1]
        else:
            probability = float(prediction)
        
        # Determine risk level and confidence
        risk_level = self._get_risk_level(probability)
        confidence = self._get_confidence(probability)
        message = self._get_recommendation(prediction, probability, risk_level)
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease',
            'probability': round(float(probability), 4),
            'risk_level': risk_level,
            'confidence': confidence,
            'message': message
        }
        
        return result
    
    def _get_risk_level(self, probability: float) -> str:
        """
        Determine risk level based on probability.
        """
        if probability < 0.25:
            return "Low"
        elif probability < 0.50:
            return "Medium"
        elif probability < 0.75:
            return "High"
        else:
            return "Very High"
    
    def _get_confidence(self, probability: float) -> str:
        """
        Determine model confidence based on probability distance from 0.5.
        """
        distance = abs(probability - 0.5)
        
        if distance > 0.4:
            return "Very High"
        elif distance > 0.3:
            return "High"
        elif distance > 0.2:
            return "Medium"
        else:
            return "Low"
    
    def _get_recommendation(self, prediction: int, probability: float, risk_level: str) -> str:
        """
        Generate recommendation message based on prediction.
        """
        if prediction == 0:
            if probability < 0.25:
                return ("Low risk of heart disease detected. Continue maintaining a healthy lifestyle "
                       "with regular exercise and balanced diet. Routine check-ups are recommended.")
            else:
                return ("No immediate heart disease indicators, but some risk factors are present. "
                       "Consider lifestyle modifications and schedule a follow-up with your physician.")
        else:
            if probability > 0.75:
                return ("High probability of heart disease detected. Immediate consultation with a "
                       "cardiologist is strongly recommended. Please seek medical attention promptly.")
            else:
                return ("Moderate to high risk of heart disease indicated. Please consult a healthcare "
                       "professional for comprehensive cardiac evaluation and further testing.")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model (if available).
        """
        if self.model is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return {}
        
        return dict(zip(self.feature_names, importances))


# ============================================
# TEST PREDICTION (FOR DEVELOPMENT)
# ============================================
if __name__ == "__main__":
    # Test the predictor
    print("\n" + "=" * 50)
    print("TESTING HEART DISEASE PREDICTOR")
    print("=" * 50)
    
    # Initialize predictor
    predictor = HeartDiseasePredictor()
    
    # Sample patient data
    sample_patient = {
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
    
    print("\n📋 Sample Patient Data:")
    for key, value in sample_patient.items():
        print(f"   • {key}: {value}")
    
    # Make prediction
    if predictor.model is not None:
        result = predictor.predict(sample_patient)
        
        print("\n🔮 Prediction Result:")
        print(f"   • Prediction: {result['prediction_label']}")
        print(f"   • Probability: {result['probability']:.2%}")
        print(f"   • Risk Level: {result['risk_level']}")
        print(f"   • Confidence: {result['confidence']}")
        print(f"\n💬 Recommendation:")
        print(f"   {result['message']}")
    else:
        print("\n⚠️ Cannot test: Model not loaded")
