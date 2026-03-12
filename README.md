![Attrition](https://github.com/user-attachments/assets/ef778c16-6eba-45dc-9bd8-39560208a599)

**HR Employee Attrition Prediction & Analytics**

This repository contains an end-to-end machine learning solution designed to help Human Resource departments understand and predict employee turnover. By analyzing key workplace factors, the project provides actionable insights to improve retention strategies.

**Project Overview**
The core of this project is a Random Forest Classifier trained to identify patterns that lead to employee attrition. The solution includes a comprehensive data analysis pipeline and a user-friendly web interface built with Streamlit for real-time risk assessment.

**Key Components:**
Data Analysis: Detailed Exploratory Data Analysis (EDA) in Jupyter Notebooks to uncover correlations between tenure, satisfaction, and turnover.
Predictive Modeling: A machine learning pipeline featuring data preprocessing, feature encoding, and model evaluation.
Interactive Dashboard: A professional Streamlit application that allows users to input employee metrics and receive instant attrition predictions.
Business Intelligence: Visualizations highlighting the "Top 10 Important Features" that drive employee decisions to leave.

**Tech Stack**
Language: Python 3.x
Data Science: Pandas, NumPy, Scikit-Learn
Visualization: Matplotlib, Seaborn
Web Framework: Streamlit
Deployment: Pickle (for model serialization)

**Project Structure**

├── src/                    # Modular source code
│   ├── preprocessing.py    # Data cleaning and transformation
│   ├── model_training.py   # Training logic for Random Forest
│   ├── prediction.py       # Inference engine
│   └── visualization.py    # Dashboard plotting functions
├── EDA_and_Modelling.ipynb # Experimentation and feature engineering
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
└── README.md

**Installation & Usage**
Clone the repository:
git clone https://github.com/HammadRana01/hr-attrition-project.git
cd hr-attrition-project

**Install dependencies:**
pip install -r requirements.txt

**Run the application:**
streamlit run app.py

**Model Performance**
The model utilizes a Random Forest algorithm to evaluate over 30 features. It specifically focuses on high-impact variables such as:
OverTime: The strongest indicator of attrition.
MonthlyIncome: Financial stability and its role in retention.
JobSatisfaction & EnvironmentSatisfaction: The impact of workplace culture.
The system provides a clear output: Low Attrition Risk (with engagement recommendations) or High Attrition Risk (suggesting immediate intervention).

**Future Improvements**
Implementing SHAP values for deeper model transparency.
Adding support for more complex algorithms like XGBoost or LightGBM.
Integrating automated retuning of the model as new data arrives.
Created for HR Analytics & Strategic Decision Making.
