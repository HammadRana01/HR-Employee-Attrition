# src/preprocessing.py

import pandas as pd

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Drop irrelevant columns
    df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)

    # Encode target label
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    # One-hot encode categorical features
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df
