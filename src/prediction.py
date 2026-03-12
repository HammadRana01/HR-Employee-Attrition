# src/prediction.py

import pandas as pd

def get_prediction(input_dict, model, feature_names):
    # Convert input_dict to a single-row DataFrame with proper column order
    input_df = pd.DataFrame([input_dict], columns=feature_names)

    # Ensure the column names match what model expects
    missing_cols = set(feature_names) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    input_df = input_df[feature_names]

    prediction = model.predict(input_df)[0]
    return prediction
