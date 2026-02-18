# ❤️ Heart Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AI-Powered Heart Disease Prediction System using Machine Learning Classification Algorithms**

An end-to-end machine learning project that predicts the likelihood of heart disease based on clinical and demographic patient data. Built with FastAPI backend, Streamlit frontend, and multiple ML classification models.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Deployment](#-deployment)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a complete machine learning pipeline for heart disease detection:

- **Problem Type:** Supervised Learning - Binary Classification
- **Goal:** Predict whether a patient has heart disease (0 = No, 1 = Yes)
- **Models Trained:** Logistic Regression, Decision Tree, Random Forest, SVM
- **Best Model:** Selected based on ROC-AUC score
- **Application:** Web-based prediction system with REST API

### Key Objectives

1. Train and compare multiple ML classification algorithms
2. Evaluate models using comprehensive metrics (Recall, Precision, F1, ROC-AUC)
3. Deploy a production-ready prediction system
4. Provide an intuitive user interface for healthcare professionals

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Heart Disease Detection System               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     │
│  │   Frontend   │────▶│   Backend    │────▶│    Model     │     │
│  │  (Streamlit) │     │  (FastAPI)   │     │   (.pkl)     │     │
│  └──────────────┘     └──────────────┘     └──────────────┘     │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     │
│  │  User Input  │     │  REST API    │     │  Prediction  │     │
│  │    Form      │     │  /predict    │     │    Engine    │     │
│  └──────────────┘     └──────────────┘     └──────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Data Flow:
1. User enters patient data in Streamlit UI
2. Frontend sends POST request to FastAPI backend
3. Backend preprocesses data and loads trained model
4. Model makes prediction with probability score
5. Results returned to user with risk assessment
```

---

## ✨ Features

### Machine Learning
- ✅ Multiple classification algorithms (LR, DT, RF, SVM)
- ✅ Automatic model comparison and selection
- ✅ Feature scaling with StandardScaler
- ✅ Comprehensive evaluation metrics
- ✅ Confusion matrix and ROC curve visualization
- ✅ Feature importance analysis

### Backend (API)
- ✅ FastAPI REST API
- ✅ Pydantic data validation
- ✅ Swagger/OpenAPI documentation
- ✅ CORS middleware support
- ✅ Health check endpoints
- ✅ Error handling

### Frontend (UI)
- ✅ Modern Streamlit interface
- ✅ Interactive input forms
- ✅ Real-time predictions
- ✅ Probability visualization
- ✅ Risk level assessment
- ✅ Medical recommendations

### Deployment
- ✅ Docker support
- ✅ Render deployment ready
- ✅ Environment configuration
- ✅ Production-ready architecture

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.9+ |
| **ML Framework** | scikit-learn |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Serialization** | Joblib |
| **Containerization** | Docker |
| **Deployment** | Render |

---

## 📁 Project Structure

```
heart_disease_detection/
│
├── frontend/                   # Streamlit Frontend
│   ├── app.py                  # Main Streamlit application
│   └── requirements.txt        # Frontend dependencies
│
├── backend/                    # FastAPI Backend
│   ├── main.py                 # API entry point
│   ├── predict.py              # Prediction logic
│   ├── preprocess.py           # Data preprocessing
│   └── requirements.txt        # Backend dependencies
│
├── model/                      # Trained Models
│   ├── model.pkl               # Best trained model
│   ├── scaler.pkl              # Feature scaler
│   ├── feature_names.pkl       # Feature names
│   └── model_info.pkl          # Model metadata
│
├── training/                   # Training Scripts
│   ├── train.py                # Model training script
│   └── requirements.txt        # Training dependencies
│
├── dataset/                    # Dataset
│   └── heart_disease_dataset.csv
│
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Master dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore rules
```

### Folder Descriptions

| Folder | Purpose |
|--------|---------|
| `frontend/` | Streamlit web application for user interaction |
| `backend/` | FastAPI server handling predictions |
| `model/` | Trained ML models and preprocessing objects |
| `training/` | Scripts for model training and evaluation |
| `dataset/` | Heart disease dataset (CSV) |

---

## 📊 Dataset

### Overview
- **Source:** Heart Disease Dataset
- **Samples:** 401 patients
- **Features:** 13 clinical parameters
- **Target:** Heart disease presence (binary)

### Features

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `age` | Numeric | Age in years |
| 2 | `sex` | Binary | 0=Female, 1=Male |
| 3 | `chest_pain_type` | Categorical | 0-3 (4 types) |
| 4 | `resting_blood_pressure` | Numeric | mm Hg |
| 5 | `cholesterol` | Numeric | mg/dl |
| 6 | `fasting_blood_sugar` | Binary | >120 mg/dl |
| 7 | `resting_ecg` | Categorical | 0-2 (3 types) |
| 8 | `max_heart_rate` | Numeric | bpm |
| 9 | `exercise_induced_angina` | Binary | 0=No, 1=Yes |
| 10 | `st_depression` | Numeric | 0-10 |
| 11 | `st_slope` | Categorical | 0-2 (3 types) |
| 12 | `num_major_vessels` | Numeric | 0-4 |
| 13 | `thalassemia` | Categorical | 0-3 (4 types) |

### Target Variable
- `heart_disease`: 0 = No Disease, 1 = Disease

---

## 📈 Model Performance

### Algorithms Compared

1. **Logistic Regression** - Linear classification
2. **Decision Tree** - Tree-based classification
3. **Random Forest** - Ensemble of decision trees
4. **Support Vector Machine** - Kernel-based classification

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Recall** | Sensitivity - ability to detect disease |
| **Precision** | Positive predictive value |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under ROC curve |
| **Accuracy** | Overall correct predictions |
| **Specificity** | True negative rate |

### Model Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.85 | ~0.84 | ~0.86 | ~0.85 | ~0.90 |
| Decision Tree | ~0.78 | ~0.77 | ~0.80 | ~0.78 | ~0.78 |
| Random Forest | ~0.87 | ~0.86 | ~0.88 | ~0.87 | ~0.92 |
| SVM | ~0.84 | ~0.83 | ~0.85 | ~0.84 | ~0.89 |

*Note: Actual values may vary based on random state and data split*

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git

### Clone Repository

```bash
git clone https://github.com/yourusername/heart-disease-detection.git
cd heart-disease-detection
```

### Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install separately
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

---

## 📖 Usage

### Step 1: Train Model (Google Colab)

1. Open Google Colab: https://colab.research.google.com
2. Upload `training/train.py` and `dataset/heart_disease_dataset.csv`
3. Run the training script:

```python
# In Google Colab
!pip install scikit-learn pandas matplotlib seaborn joblib
!python train.py
```

4. Download the generated files from `model/` folder:
   - `model.pkl`
   - `scaler.pkl`
   - `feature_names.pkl`
   - `model_info.pkl`

5. Place them in your project's `model/` directory

### Step 2: Start Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API will be available at: http://localhost:8000
Swagger docs at: http://localhost:8000/docs

### Step 3: Start Frontend

```bash
cd frontend
streamlit run app.py --server.port 8501
```

App will be available at: http://localhost:8501

### Step 4: Make Predictions

1. Open the Streamlit app in your browser
2. Enter patient clinical data
3. Click "Predict"
4. View the results and recommendations

---

## 📡 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | API health check |
| GET | `/model-info` | Model information |
| GET | `/features` | Feature descriptions |
| POST | `/predict` | Make prediction |

### Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Prediction Response

```json
{
  "prediction": 1,
  "prediction_label": "Heart Disease Detected",
  "probability": 0.78,
  "risk_level": "High",
  "confidence": "High",
  "message": "High probability of heart disease detected..."
}
```

---

## ☁️ Deployment

### Deploy to Render

#### Backend Deployment

1. Push code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Create new "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name:** heart-disease-backend
   - **Root Directory:** backend
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

#### Frontend Deployment

1. Create new "Web Service" on Render
2. Connect same repository
3. Configure:
   - **Name:** heart-disease-frontend
   - **Root Directory:** frontend
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Set environment variable:
   - `API_URL`: Your backend URL

### Docker Deployment

```bash
# Build image
docker build -t heart-disease-backend ./backend

# Run container
docker run -d -p 8000:8000 heart-disease-backend
```

---

## 📸 Screenshots

### Streamlit Frontend
![Frontend Screenshot](screenshots/frontend.png)
*Patient data input form with prediction results*

### API Documentation
![API Docs Screenshot](screenshots/api_docs.png)
*FastAPI Swagger documentation*

### Model Comparison
![Model Comparison](screenshots/model_comparison.png)
*Performance comparison of all trained models*

### ROC Curves
![ROC Curves](screenshots/roc_curves.png)
*ROC curves for model evaluation*

---

## 🔒 Disclaimer

⚠️ **Medical Disclaimer:**

This application is developed for **educational and demonstration purposes only**. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.

- Always seek the advice of qualified healthcare providers
- The predictions are based on machine learning models with inherent limitations
- Do not make medical decisions based solely on this application's output

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Heart Disease Detection Team**

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset inspiration
- FastAPI and Streamlit communities
- scikit-learn documentation and tutorials

---

## 📧 Contact

For questions or support, please open an issue on GitHub or contact:
- Email: support@heartdiseasedetection.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

**⭐ If you found this project useful, please give it a star!**
