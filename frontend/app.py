"""
Heart Disease Detection - Streamlit Frontend
=============================================
Professional web interface for heart disease prediction.

Author: Heart Disease Detection Project
Date: 2024
"""

import streamlit as st
import requests
import json
import time

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Heart Disease Detection",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .result-positive {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
    }
    
    .result-negative {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
    }
    
    /* Risk level badges */
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        margin: 0.5rem;
    }
    
    .risk-low { background-color: #51cf66; color: white; }
    .risk-medium { background-color: #fcc419; color: #333; }
    .risk-high { background-color: #ff922b; color: white; }
    .risk-very-high { background-color: #ff6b6b; color: white; }
    
    /* Form styling */
    .stSelectbox, .stNumberInput {
        margin-bottom: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        width: 100%;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
    }
    
    /* Info cards */
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Progress bar */
    .probability-bar {
        height: 30px;
        border-radius: 15px;
        background: #e9ecef;
        overflow: hidden;
    }
    
    .probability-fill {
        height: 100%;
        border-radius: 15px;
        transition: width 0.5s ease;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# API CONFIGURATION
# ============================================
# Change this URL based on your deployment
API_URL = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8000",
    help="Backend API URL"
)


# ============================================
# HELPER FUNCTIONS
# ============================================
def make_prediction(data: dict) -> dict:
    """
    Make prediction request to the backend API.
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.json().get("detail", "Unknown error")}
    
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to the API. Please ensure the backend is running."}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. Please try again."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def check_api_health() -> bool:
    """
    Check if the API is healthy.
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_risk_color(risk_level: str) -> str:
    """
    Get color based on risk level.
    """
    colors = {
        "Low": "#51cf66",
        "Medium": "#fcc419",
        "High": "#ff922b",
        "Very High": "#ff6b6b"
    }
    return colors.get(risk_level, "#6c757d")


# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """
    Main application function.
    """
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="header-title">❤️ Heart Disease Detection</div>
        <div class="header-subtitle">AI-Powered Cardiac Risk Assessment Tool</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown("""
        This application uses machine learning to predict the likelihood of heart disease 
        based on clinical parameters.
        
        **How to use:**
        1. Enter patient information
        2. Click 'Predict'
        3. View results and recommendations
        
        ⚠️ **Disclaimer:** This tool is for educational purposes only. 
        Always consult healthcare professionals for medical decisions.
        """)
        
        st.markdown("---")
        
        # API Status
        st.markdown("## 🔌 API Status")
        if check_api_health():
            st.success("✅ Connected")
        else:
            st.error("❌ Disconnected")
            st.info("Make sure the backend is running:\n```\nuvicorn main:app --reload\n```")
        
        st.markdown("---")
        
        st.markdown("## 📊 Model Info")
        st.markdown("""
        - **Algorithm:** Classification ML
        - **Features:** 13 clinical parameters
        - **Accuracy:** ~85%+
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 Patient Information")
        
        # Create form for patient data
        with st.form("patient_form"):
            # Row 1: Age and Sex
            col_a, col_b = st.columns(2)
            with col_a:
                age = st.number_input(
                    "Age",
                    min_value=1,
                    max_value=120,
                    value=55,
                    help="Patient's age in years"
                )
            with col_b:
                sex = st.selectbox(
                    "Sex",
                    options=[("Male", 1), ("Female", 0)],
                    format_func=lambda x: x[0],
                    help="Patient's gender"
                )
            
            # Row 2: Chest Pain and Blood Pressure
            col_c, col_d = st.columns(2)
            with col_c:
                chest_pain_type = st.selectbox(
                    "Chest Pain Type",
                    options=[
                        ("Typical Angina", 0),
                        ("Atypical Angina", 1),
                        ("Non-anginal Pain", 2),
                        ("Asymptomatic", 3)
                    ],
                    format_func=lambda x: x[0],
                    help="Type of chest pain experienced"
                )
            with col_d:
                resting_bp = st.number_input(
                    "Resting Blood Pressure (mm Hg)",
                    min_value=50,
                    max_value=300,
                    value=130,
                    help="Blood pressure on admission"
                )
            
            # Row 3: Cholesterol and Fasting Blood Sugar
            col_e, col_f = st.columns(2)
            with col_e:
                cholesterol = st.number_input(
                    "Cholesterol (mg/dl)",
                    min_value=100,
                    max_value=600,
                    value=250,
                    help="Serum cholesterol level"
                )
            with col_f:
                fasting_bs = st.selectbox(
                    "Fasting Blood Sugar > 120 mg/dl",
                    options=[("No", 0), ("Yes", 1)],
                    format_func=lambda x: x[0],
                    help="Is fasting blood sugar greater than 120 mg/dl?"
                )
            
            # Row 4: Resting ECG and Max Heart Rate
            col_g, col_h = st.columns(2)
            with col_g:
                resting_ecg = st.selectbox(
                    "Resting ECG",
                    options=[
                        ("Normal", 0),
                        ("ST-T Wave Abnormality", 1),
                        ("Left Ventricular Hypertrophy", 2)
                    ],
                    format_func=lambda x: x[0],
                    help="Resting electrocardiographic results"
                )
            with col_h:
                max_hr = st.number_input(
                    "Max Heart Rate (bpm)",
                    min_value=60,
                    max_value=250,
                    value=150,
                    help="Maximum heart rate achieved during exercise"
                )
            
            # Row 5: Exercise Angina and ST Depression
            col_i, col_j = st.columns(2)
            with col_i:
                exercise_angina = st.selectbox(
                    "Exercise Induced Angina",
                    options=[("No", 0), ("Yes", 1)],
                    format_func=lambda x: x[0],
                    help="Angina induced by exercise"
                )
            with col_j:
                st_depression = st.number_input(
                    "ST Depression",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.5,
                    step=0.1,
                    help="ST depression induced by exercise"
                )
            
            # Row 6: ST Slope and Major Vessels
            col_k, col_l = st.columns(2)
            with col_k:
                st_slope = st.selectbox(
                    "ST Slope",
                    options=[
                        ("Upsloping", 0),
                        ("Flat", 1),
                        ("Downsloping", 2)
                    ],
                    format_func=lambda x: x[0],
                    help="Slope of peak exercise ST segment"
                )
            with col_l:
                num_vessels = st.selectbox(
                    "Number of Major Vessels",
                    options=[0, 1, 2, 3, 4],
                    index=1,
                    help="Number of major vessels colored by fluoroscopy"
                )
            
            # Row 7: Thalassemia
            thalassemia = st.selectbox(
                "Thalassemia",
                options=[
                    ("Normal", 0),
                    ("Fixed Defect", 1),
                    ("Reversible Defect", 2),
                    ("Not Described", 3)
                ],
                format_func=lambda x: x[0],
                help="Thalassemia blood disorder type"
            )
            
            # Submit button
            submitted = st.form_submit_button(
                "🔮 Predict Heart Disease Risk",
                use_container_width=True
            )
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        if submitted:
            # Prepare data for API
            patient_data = {
                "age": age,
                "sex": sex[1],
                "chest_pain_type": chest_pain_type[1],
                "resting_blood_pressure": resting_bp,
                "cholesterol": cholesterol,
                "fasting_blood_sugar": fasting_bs[1],
                "resting_ecg": resting_ecg[1],
                "max_heart_rate": max_hr,
                "exercise_induced_angina": exercise_angina[1],
                "st_depression": st_depression,
                "st_slope": st_slope[1],
                "num_major_vessels": num_vessels,
                "thalassemia": thalassemia[1]
            }
            
            # Show loading spinner
            with st.spinner("Analyzing patient data..."):
                time.sleep(0.5)  # Brief delay for UX
                result = make_prediction(patient_data)
            
            if result["success"]:
                data = result["data"]
                prediction = data["prediction"]
                probability = data["probability"]
                risk_level = data["risk_level"]
                
                # Display result card
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-card result-positive">
                        <h2 style="margin:0;">⚠️ Heart Disease Detected</h2>
                        <p style="font-size:1.2rem; margin:0.5rem 0;">Risk Level: <strong>{risk_level}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card result-negative">
                        <h2 style="margin:0;">✅ No Heart Disease Detected</h2>
                        <p style="font-size:1.2rem; margin:0.5rem 0;">Risk Level: <strong>{risk_level}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability meter
                st.markdown("#### 📈 Disease Probability")
                prob_percentage = probability * 100
                
                # Color based on probability
                if prob_percentage < 25:
                    bar_color = "#51cf66"
                elif prob_percentage < 50:
                    bar_color = "#fcc419"
                elif prob_percentage < 75:
                    bar_color = "#ff922b"
                else:
                    bar_color = "#ff6b6b"
                
                st.progress(probability)
                st.markdown(f"<h3 style='text-align:center; color:{bar_color};'>{prob_percentage:.1f}%</h3>", 
                           unsafe_allow_html=True)
                
                # Additional metrics
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Confidence", data["confidence"])
                with col_m2:
                    st.metric("Risk Level", risk_level)
                
                # Recommendation
                st.markdown("#### 💬 Recommendation")
                st.info(data["message"])
                
                # Medical disclaimer
                st.warning("""
                ⚠️ **Important Disclaimer:**
                
                This prediction is generated by a machine learning model and should NOT be used as a 
                substitute for professional medical advice, diagnosis, or treatment. Always seek the 
                advice of your physician or other qualified health provider with any questions you may 
                have regarding a medical condition.
                """)
                
            else:
                st.error(f"❌ Prediction failed: {result['error']}")
                st.info("""
                **Troubleshooting:**
                1. Ensure the backend server is running
                2. Check if the API URL is correct
                3. Verify the model files are in place
                """)
        
        else:
            # Default state
            st.markdown("""
            <div class="info-card">
                <h4>👋 Welcome!</h4>
                <p>Enter patient information on the left and click <strong>Predict</strong> 
                to get heart disease risk assessment.</p>
                <p><em>All fields are required for accurate prediction.</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature importance info
            with st.expander("ℹ️ Understanding the Features"):
                st.markdown("""
                | Feature | Description |
                |---------|-------------|
                | **Age** | Patient's age in years |
                | **Sex** | Male or Female |
                | **Chest Pain Type** | Type of chest pain (4 categories) |
                | **Resting Blood Pressure** | Blood pressure when resting (mm Hg) |
                | **Cholesterol** | Serum cholesterol level (mg/dl) |
                | **Fasting Blood Sugar** | Blood sugar > 120 mg/dl (Yes/No) |
                | **Resting ECG** | Electrocardiogram results (3 categories) |
                | **Max Heart Rate** | Maximum heart rate during exercise |
                | **Exercise Induced Angina** | Chest pain during exercise (Yes/No) |
                | **ST Depression** | ST segment depression during exercise |
                | **ST Slope** | Slope of ST segment (3 categories) |
                | **Major Vessels** | Vessels colored by fluoroscopy (0-4) |
                | **Thalassemia** | Blood disorder type (4 categories) |
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>❤️ Heart Disease Detection System | Built with Streamlit & FastAPI</p>
        <p>For educational purposes only. Always consult healthcare professionals.</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================
# RUN APPLICATION
# ============================================
if __name__ == "__main__":
    main()
