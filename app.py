from sklearn.ensemble import GradientBoostingRegressor


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import Random ForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px

# Set the path to the folder containing your model files
model_path = "model_files"  # Change this to the correct folder path

# Define model file name
model_file = os.path.join(model_path, "best_model.joblib")

# Load the primary model or fallback to a default if not found
try:
    model = joblib.load(model_file)  # Load the specified model
except FileNotFoundError:
    st.warning(f"Model file '{model_file}' not found. Loading a default DecisionTreeRegressor as fallback.")
    model = DecisionTreeRegressor()  # Default model if the primary one is not found

# Load the encoder and scaler files
try:
    encoder = joblib.load(os.path.join(model_path, "encoder.joblib"))
    scaler = joblib.load(os.path.join(model_path, "scaler.joblib"))
except FileNotFoundError:
    st.error("Encoder or Scaler file not found. Ensure the correct paths and files exist.")

# Continue with the Streamlit app logic
st.title("Grocery Store Forecasting App")

city = st.text_input("City")
store_id = st.text_input("Store ID")
onpromotion = st.selectbox("On Promotion?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
is_holiday = st.selectbox("Holiday?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
category_id = st.text_input("Category ID")

# Generate the forecast if all inputs are provided
if st.button("Generate Forecast"):
    if not city or not store_id or not category_id:
        st.warning("Please provide all required inputs.")
    else:
        # Prepare the input data
        input_data = pd.DataFrame({
            "city": [city],
            "store_id": [store_id],
            "onpromotion": [onpromotion],
            "is_holiday": [is_holiday],
            "category_id": [category_id]
        })

        # Apply encoder and scaler (make sure they were loaded correctly)
        if 'encoder' in locals() and 'scaler' in locals():
            encoded_input = encoder.transform(input_data)
            scaled_input = scaler.transform(encoded_input)

            # Make predictions
            forecast = model.predict(scaled_input)

            # Display forecast results
            st.header("Forecast for the next eight weeks:")
            forecast_df = pd.DataFrame({
                "Week": np.arange(1, 9),
                "Predicted Sales": forecast.flatten()  # Flatten if prediction is 2D
            })
            st.table(forecast_df)

            # Visualize the forecast with a plot
            st.plotly_chart(px.line(forecast_df, x="Week", y="Predicted Sales", title="Forecasted Sales for the Next Eight Weeks"))
