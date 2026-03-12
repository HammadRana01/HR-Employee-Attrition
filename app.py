# app.py

import streamlit as st
import pickle
from src.preprocessing import load_and_preprocess_data
from src.model_training import train_model
from src.prediction import get_prediction
from src.visualization import display_visuals

# Streamlit Page Config
st.set_page_config(
    page_title="HR Attrition Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
        /* Main background and container styling */
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 0;
        }
        
        .block-container {
            padding: 2rem 3rem 2rem 3rem;
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            margin: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Text styling - all black */
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: #000000 !important;
        }
        
        /* Header styling */
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        .main-header h1 {
            color: white !important;
            font-size: 3rem;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .main-header p {
            color: white !important;
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        /* Card styling */
        .info-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
            border-left: 4px solid #667eea;
        }
        
        .prediction-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        .prediction-card h3, .prediction-card p {
            color: white !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            font-weight: bold;
            border-radius: 12px;
            padding: 12px 24px;
            border: none;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }
        
        /* Metric styling */
        .metric-container {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            margin: 0.5rem 0;
        }
        
        /* Form styling */
        .stSelectbox > div > div {
            background-color: white;
            border-radius: 8px;
        }
        
        .stSlider > div > div {
            background-color: white;
            border-radius: 8px;
        }
        
        .stNumberInput > div > div {
            background-color: white;
            border-radius: 8px;
        }
        
        /* Image styling */
        .header-image {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .fade-in {
            animation: fadeIn 0.6s ease-out;
        }
    </style>
""", unsafe_allow_html=True)

# Main Header with enhanced styling
st.markdown("""
    <div class="main-header fade-in">
        <h1>🧠 HR Employee Attrition Predictor</h1>
        <p>Powered by Machine Learning • Predict Employee Retention with Confidence</p>
    </div>
""", unsafe_allow_html=True)

# Load and preprocess data
try:
    df = load_and_preprocess_data("data/HR-Employee-Attrition.csv")
    # Train/load model
    model, feature_names = train_model(df)
    model_loaded = True
except:
    st.error("⚠️ Could not load the HR dataset. Please ensure 'data/HR-Employee-Attrition.csv' exists.")
    model_loaded = False

# Enhanced Sidebar Navigation
st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white !important; margin: 0;">📂 Navigation</h3>
    </div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("", ["🏠 Home", "📊 Visualizations", "🧾 Predict Attrition"], index=0)

# Add sidebar info
st.sidebar.markdown("""
    <div class="info-card">
        <h4>📈 Model Performance</h4>
        <div class="metric-container">
            <strong>Algorithm:</strong> Random Forest<br>
            <strong>Accuracy:</strong> ~85%<br>
            <strong>Features:</strong> 30+
        </div>
    </div>
""", unsafe_allow_html=True)

# Home Section
if page == "🏠 Home":
    # Hero image
    st.markdown("""
        <img src="https://images.unsplash.com/photo-1552664730-d307ca884978?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" 
             class="header-image" alt="HR Analytics">
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-card fade-in">
            <h2>👋 Welcome to HR Attrition Predictor!</h2>
            <p style="font-size: 1.1rem; line-height: 1.6;">
                This intelligent application leverages advanced Machine Learning algorithms to predict employee attrition 
                based on comprehensive HR data analysis. Our model analyzes multiple factors including demographics, 
                job satisfaction, compensation, and work environment to provide accurate predictions.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="info-card fade-in">
                <div class="feature-icon">📊</div>
                <h3>Data Analytics</h3>
                <p>Comprehensive visualization of HR data patterns and trends to understand employee behavior.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-card fade-in">
                <div class="feature-icon">🤖</div>
                <h3>ML Predictions</h3>
                <p>State-of-the-art Random Forest algorithm trained on historical HR data for accurate predictions.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="info-card fade-in">
                <div class="feature-icon">📈</div>
                <h3>Business Insights</h3>
                <p>Actionable insights to help HR teams make data-driven decisions about employee retention.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # How it works section
    st.markdown("""
        <div class="info-card fade-in">
            <h2>🔍 How It Works</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📝</div>
                    <h4>1. Input Data</h4>
                    <p>Enter employee information including age, income, experience, and work patterns.</p>
                </div>
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚙️</div>
                    <h4>2. ML Processing</h4>
                    <p>Our trained Random Forest model analyzes the data against historical patterns.</p>
                </div>
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                    <h4>3. Get Results</h4>
                    <p>Receive instant predictions with confidence scores and actionable insights.</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Visualizations Section
elif page == "📊 Visualizations":
    if model_loaded:
        st.markdown("""
            <img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" 
                 class="header-image" alt="Data Visualization">
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-card fade-in">
                <h2>📈 HR Data Insights & Analytics</h2>
                <p style="font-size: 1.1rem;">
                    Explore comprehensive visualizations of our HR dataset to understand patterns, 
                    trends, and key factors influencing employee attrition.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        display_visuals(df)
    else:
        st.error("Cannot display visualizations without data.")

# Prediction Section
elif page == "🧾 Predict Attrition":
    if model_loaded:
        st.markdown("""
            <img src="https://images.unsplash.com/photo-1460925895917-afdab827c52f?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" 
                 class="header-image" alt="Prediction Analytics">
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="prediction-card fade-in">
                <h2>🧾 Employee Attrition Prediction</h2>
                <p>Enter the employee details below to get an instant prediction about their likelihood to leave the company.</p>
            </div>
        """, unsafe_allow_html=True)

        # Create input form in columns
        col1, col2 = st.columns(2)
        input_dict = {}

        with col1:
            st.markdown("""
                <div class="info-card">
                    <h3>👤 Personal Information</h3>
                </div>
            """, unsafe_allow_html=True)
            
            input_dict['Age'] = st.slider("🎂 Age", 18, 60, 30, help="Employee's current age")
            input_dict['DistanceFromHome'] = st.slider("🏠 Distance From Home (km)", 1, 50, 10, 
                                                      help="Distance between home and workplace")
            input_dict['MonthlyIncome'] = st.number_input("💰 Monthly Income (₹)", min_value=1000, max_value=50000, 
                                                         value=15000, step=500, help="Current monthly salary")
            input_dict['NumCompaniesWorked'] = st.slider("🏢 Number of Companies Worked", 0, 10, 2, 
                                                        help="Total companies worked for previously")

        with col2:
            st.markdown("""
                <div class="info-card">
                    <h3>💼 Work Experience</h3>
                </div>
            """, unsafe_allow_html=True)
            
            input_dict['TotalWorkingYears'] = st.slider("📅 Total Working Years", 0, 40, 10, 
                                                       help="Total years of professional experience")
            input_dict['YearsAtCompany'] = st.slider("🏛️ Years at Current Company", 0, 30, 5, 
                                                     help="Years spent at current company")
            overtime_selection = st.selectbox("⏰ Works Overtime?", ['No', 'Yes'], 
                                            help="Does the employee regularly work overtime?")
            input_dict['OverTime_Yes'] = overtime_selection == 'Yes'

        # Fill other feature columns with 0
        for col in feature_names:
            if col not in input_dict:
                input_dict[col] = 0

        # Enhanced Predict Button
        st.markdown("<br>", unsafe_allow_html=True)
        col_center = st.columns([1, 2, 1])
        
        with col_center[1]:
            if st.button("🔍 Predict Employee Attrition", use_container_width=True):
                with st.spinner("🤖 Analyzing employee data..."):
                    prediction = get_prediction(input_dict, model, feature_names)
                    
                    if prediction == 1:
                        st.markdown("""
                            <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                                        color: white; padding: 2rem; border-radius: 15px; text-align: center; 
                                        margin: 2rem 0; box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);">
                                <h2 style="color: white !important;">⚠️ High Attrition Risk</h2>
                                <p style="color: white !important; font-size: 1.2rem;">
                                    The employee is likely to leave the company based on the current profile.
                                </p>
                                <p style="color: white !important; font-size: 1rem; opacity: 0.9;">
                                    <strong>Recommendation:</strong> Consider retention strategies such as career development, 
                                    compensation review, or work-life balance improvements.
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div style="background: linear-gradient(135deg, #00b894 0%, #00a085 100%); 
                                        color: white; padding: 2rem; border-radius: 15px; text-align: center; 
                                        margin: 2rem 0; box-shadow: 0 4px 15px rgba(0, 184, 148, 0.3);">
                                <h2 style="color: white !important;">✅ Low Attrition Risk</h2>
                                <p style="color: white !important; font-size: 1.2rem;">
                                    The employee is likely to stay with the company based on the current profile.
                                </p>
                                <p style="color: white !important; font-size: 1rem; opacity: 0.9;">
                                    <strong>Recommendation:</strong> Continue current engagement strategies and 
                                    monitor satisfaction levels regularly.
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
    else:
        st.error("Cannot make predictions without the trained model and data.")

# Footer
st.markdown("""
    <div style="text-align: center; padding: 2rem; margin-top: 3rem; 
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; color: white;">
        <p style="color: white !important; margin: 0; font-size: 1rem;">
            🧠 HR Attrition Predictor | Powered by Machine Learning & Data Science
        </p>
        <p style="color: white !important; margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.8;">
            Making HR decisions smarter, one prediction at a time
        </p>
    </div>
""", unsafe_allow_html=True)