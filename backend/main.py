"""
Heart Disease Detection - FastAPI Backend
==========================================
Main API entry point for heart disease prediction service.

Author: Heart Disease Detection Project
Date: 2024
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional
import uvicorn

# Import prediction logic
from predict import HeartDiseasePredictor

# ============================================
# FASTAPI APP INITIALIZATION
# ============================================
app = FastAPI(
    title="Heart Disease Detection API",
    description="""
    ## Heart Disease Prediction API
    
    This API predicts the likelihood of heart disease based on patient clinical data.
    
    ### Features:
    - Real-time heart disease prediction
    - Probability scores for risk assessment
    - Support for all clinical input parameters
    
    ### Model Information:
    - Algorithm: Machine Learning Classification
    - Accuracy: ~85%+ (varies by model)
    - Input: 13 clinical features
    - Output: Prediction (0/1) + Probability
    
    ⚠️ **Disclaimer**: This is for educational purposes only. 
    Always consult healthcare professionals for medical decisions.
    """,
    version="1.0.0",
    contact={
        "name": "Heart Disease Detection Team",
        "email": "support@heartdiseasedetection.com"
    }
)

# ============================================
# CORS MIDDLEWARE
# ============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE
# ============================================
class PatientData(BaseModel):
    """
    Input schema for patient clinical data.
    All fields are required for accurate prediction.
    """
    age: int = Field(
        ..., 
        ge=1, 
        le=120, 
        description="Patient's age in years (1-120)"
    )
    sex: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Gender: 0 = Female, 1 = Male"
    )
    chest_pain_type: int = Field(
        ..., 
        ge=0, 
        le=3, 
        description="Chest pain type: 0=Typical Angina, 1=Atypical Angina, 2=Non-anginal Pain, 3=Asymptomatic"
    )
    resting_blood_pressure: int = Field(
        ..., 
        ge=50, 
        le=300, 
        description="Resting blood pressure in mm Hg (50-300)"
    )
    cholesterol: int = Field(
        ..., 
        ge=100, 
        le=600, 
        description="Serum cholesterol in mg/dl (100-600)"
    )
    fasting_blood_sugar: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Fasting blood sugar > 120 mg/dl: 0 = No, 1 = Yes"
    )
    resting_ecg: int = Field(
        ..., 
        ge=0, 
        le=2, 
        description="Resting ECG results: 0=Normal, 1=ST-T wave abnormality, 2=Left ventricular hypertrophy"
    )
    max_heart_rate: int = Field(
        ..., 
        ge=60, 
        le=250, 
        description="Maximum heart rate achieved (60-250)"
    )
    exercise_induced_angina: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Exercise induced angina: 0 = No, 1 = Yes"
    )
    st_depression: float = Field(
        ..., 
        ge=0.0, 
        le=10.0, 
        description="ST depression induced by exercise relative to rest (0-10)"
    )
    st_slope: int = Field(
        ..., 
        ge=0, 
        le=2, 
        description="Slope of peak exercise ST segment: 0=Upsloping, 1=Flat, 2=Downsloping"
    )
    num_major_vessels: int = Field(
        ..., 
        ge=0, 
        le=4, 
        description="Number of major vessels colored by fluoroscopy (0-4)"
    )
    thalassemia: int = Field(
        ..., 
        ge=0, 
        le=3, 
        description="Thalassemia: 0=Normal, 1=Fixed Defect, 2=Reversible Defect, 3=Not Described"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 55,
                "sex": 1,
                "chest_pain_type": 2,
                "resting_blood_pressure": 130,
                "cholesterol": 250,
                "fasting_blood_sugar": 0,
                "resting_ecg": 1,
                "max_heart_rate": 150,
                "exercise_induced_angina": 0,
                "st_depression": 1.5,
                "st_slope": 1,
                "num_major_vessels": 1,
                "thalassemia": 2
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    Output schema for prediction results.
    """
    prediction: int = Field(
        ..., 
        description="Prediction result: 0 = No Disease, 1 = Disease"
    )
    prediction_label: str = Field(
        ..., 
        description="Human-readable prediction label"
    )
    probability: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Probability of heart disease (0.0 to 1.0)"
    )
    risk_level: str = Field(
        ..., 
        description="Risk level: Low, Medium, High, Very High"
    )
    confidence: str = Field(
        ..., 
        description="Model confidence level"
    )
    message: str = Field(
        ..., 
        description="Additional information or recommendations"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 1,
                "prediction_label": "Heart Disease Detected",
                "probability": 0.78,
                "risk_level": "High",
                "confidence": "High",
                "message": "The model indicates a high probability of heart disease. Please consult a cardiologist for further evaluation."
            }
        }
    }


class HealthResponse(BaseModel):
    """
    Health check response schema.
    """
    model_config = {"protected_namespaces": ()}
    
    status: str
    message: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    """
    Model information response schema.
    """
    model_config = {"protected_namespaces": ()}
    
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    features: list


# ============================================
# INITIALIZE PREDICTOR
# ============================================
predictor = HeartDiseasePredictor()


# ============================================
# API ENDPOINTS
# ============================================
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - Welcome message.
    """
    return {
        "message": "Welcome to Heart Disease Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API and model status.
    """
    return HealthResponse(
        status="healthy",
        message="API is running successfully",
        model_loaded=predictor.model is not None,
        version="1.0.0"
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model.
    """
    if predictor.model_info is None:
        raise HTTPException(
            status_code=503,
            detail="Model information not available"
        )
    
    return ModelInfoResponse(
        model_name=predictor.model_info.get('model_name', 'Unknown'),
        accuracy=predictor.model_info.get('accuracy', 0.0),
        precision=predictor.model_info.get('precision', 0.0),
        recall=predictor.model_info.get('recall', 0.0),
        f1_score=predictor.model_info.get('f1_score', 0.0),
        roc_auc=predictor.model_info.get('roc_auc', 0.0),
        features=predictor.model_info.get('features', [])
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient_data: PatientData):
    """
    Predict heart disease based on patient clinical data.
    
    - **Input**: Patient clinical data (13 features)
    - **Output**: Prediction result with probability and risk assessment
    
    ⚠️ **Important**: This prediction is for screening purposes only.
    Always consult healthcare professionals for medical decisions.
    """
    try:
        # Convert Pydantic model to dictionary
        input_data = patient_data.dict()
        
        # Get prediction from predictor
        result = predictor.predict(input_data)
        
        return PredictionResponse(
            prediction=result['prediction'],
            prediction_label=result['prediction_label'],
            probability=result['probability'],
            risk_level=result['risk_level'],
            confidence=result['confidence'],
            message=result['message']
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/features", tags=["Features"])
async def get_features():
    """
    Get information about all input features.
    """
    return {
        "features": [
            {"name": "age", "type": "int", "range": "1-120", "description": "Patient's age in years"},
            {"name": "sex", "type": "int", "values": {"0": "Female", "1": "Male"}, "description": "Patient's gender"},
            {"name": "chest_pain_type", "type": "int", "values": {"0": "Typical Angina", "1": "Atypical Angina", "2": "Non-anginal Pain", "3": "Asymptomatic"}, "description": "Type of chest pain"},
            {"name": "resting_blood_pressure", "type": "int", "range": "50-300", "unit": "mm Hg", "description": "Resting blood pressure"},
            {"name": "cholesterol", "type": "int", "range": "100-600", "unit": "mg/dl", "description": "Serum cholesterol level"},
            {"name": "fasting_blood_sugar", "type": "int", "values": {"0": "≤120 mg/dl", "1": ">120 mg/dl"}, "description": "Fasting blood sugar level"},
            {"name": "resting_ecg", "type": "int", "values": {"0": "Normal", "1": "ST-T wave abnormality", "2": "Left ventricular hypertrophy"}, "description": "Resting ECG results"},
            {"name": "max_heart_rate", "type": "int", "range": "60-250", "unit": "bpm", "description": "Maximum heart rate achieved"},
            {"name": "exercise_induced_angina", "type": "int", "values": {"0": "No", "1": "Yes"}, "description": "Exercise induced angina"},
            {"name": "st_depression", "type": "float", "range": "0-10", "description": "ST depression induced by exercise"},
            {"name": "st_slope", "type": "int", "values": {"0": "Upsloping", "1": "Flat", "2": "Downsloping"}, "description": "Slope of peak exercise ST segment"},
            {"name": "num_major_vessels", "type": "int", "range": "0-4", "description": "Number of major vessels colored by fluoroscopy"},
            {"name": "thalassemia", "type": "int", "values": {"0": "Normal", "1": "Fixed Defect", "2": "Reversible Defect", "3": "Not Described"}, "description": "Thalassemia blood disorder"}
        ],
        "total_features": 13
    }


# ============================================
# RUN APPLICATION
# ============================================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
