import streamlit as st
import numpy as np
import pickle
import json

# Load the trained model
with open('bangalore_home_price_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Load feature columns
with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

# Extract location names from columns
locations = data_columns[3:]

# Streamlit App UI
st.set_page_config(page_title="Bengaluru House Price Predictor")
st.title("Bengaluru House Price Prediction")
st.write("Predict home prices based on area, location, number of bathrooms and bedrooms (BHK).")

# User Inputs
location = st.selectbox("Select Location", sorted(locations))
total_sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, value=1000)
bhk = st.slider("BHK (Bedrooms)", 1, 10, 2)
bath = st.slider("Bathrooms", 1, 10, 2)

# Prediction Function
def predict_price(location, sqft, bath, bhk):
    try:
        x = np.zeros(len(data_columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        if location in data_columns:
            loc_index = data_columns.index(location)
            x[loc_index] = 1
        return round(model.predict([x])[0], 2)
    except:
        return None

# Predict Button
if st.button("Predict Price"):
    result = predict_price(location, total_sqft, bath, bhk)
    if result is not None:
        st.success(f"Estimated Price: ₹ {result} lakhs")
    else:
        st.error(" Error occurred during prediction. Please check inputs or model.")
