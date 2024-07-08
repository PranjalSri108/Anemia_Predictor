import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load the saved model and scaler
@st.cache_resource
def load_model():
    model = load('anemia_predictor.stoplib')
    scaler = load('scaler.joblib')
    return model, scaler

model, scaler = load_model()

# Set up the Streamlit app
st.title('Anemia Detection App')

st.write("""
This app predicts whether a person is anemic based on their blood parameters.
Please enter the required information below.
""")

# Create input fields
sex = st.selectbox('Sex', options=['M', 'F'])
red_pixel = st.number_input('% Red Pixel', min_value=0.0, max_value=100.0, value=50.0)
green_pixel = st.number_input('% Green Pixel', min_value=0.0, max_value=100.0, value=50.0)
blue_pixel = st.number_input('% Blue Pixel', min_value=0.0, max_value=100.0, value=50.0)
hb = st.number_input('Hemoglobin (Hb)', min_value=0.0, max_value=20.0, value=10.0)

# Create a prediction button
if st.button('Predict'):
    # Prepare the input data
    input_data = pd.DataFrame({
        'Sex': [sex],
        '%Red Pixel': [red_pixel],
        '%Green pixel': [green_pixel],
        '%Blue pixel': [blue_pixel],
        'Hb': [hb]
    })

    # Preprocess the input data
    input_data['Sex'] = input_data['Sex'].map({'M': 0, 'F': 1})
    input_data = input_data.astype(float)
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display the result
    st.subheader('Prediction Result:')
    if prediction[0] == 'Yes':
        st.error('The person is likely to be Anemic.')
    else:
        st.success('The person is likely to be Not Anemic.')

    # Display the input values
    st.subheader('Input Values:')
    st.write(input_data)

# Add some information about the app
st.info("""
This app uses a Random Forest Classifier trained on a dataset of blood parameters to predict anemia.
Please note that this is a simplified model and should not be used for actual medical diagnosis.
Always consult with a healthcare professional for proper medical advice and diagnosis.
""")
