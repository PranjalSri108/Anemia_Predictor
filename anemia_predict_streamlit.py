import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_model():
    model = joblib.load('anemia_predictor.stoplib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_model()

st.title('Anemia Detection App')

st.write("""
This app predicts whether a person is anemic based on their blood parameters.
Please adjust the sliders below to input the required information.
""")

col1, col2 = st.columns(2)

sex = st.selectbox('Sex', options=['M', 'F'])
with col1:
    red_pixel = st.slider('% Red Pixel', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    green_pixel = st.slider('% Green Pixel', min_value=0.0, max_value=100.0, value=50.0, step=0.1)

with col2:
    blue_pixel = st.slider('% Blue Pixel', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    hb = st.slider('Hemoglobin (Hb)', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
    
if st.button('Predict'):
    input_data = pd.DataFrame({
        'Sex': [sex],
        '%Red Pixel': [red_pixel],
        '%Green pixel': [green_pixel],
        '%Blue pixel': [blue_pixel],
        'Hb': [hb]
    })
    
    input_data['Sex'] = input_data['Sex'].map({'M': 0, 'F': 1})
    input_data = input_data.astype(float)
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)

    st.subheader('Prediction Result:')
    if prediction[0] == 'Yes':
        st.error('The person is likely to be Anemic.')
    else:
        st.success('The person is likely to be Not Anemic.')
        
    st.subheader('Input Values:')
    st.write(input_data)
    
st.info("""
This app uses a Random Forest Classifier trained on a dataset of blood parameters to predict anemia.
Please note that this is a simplified model and should not be used for actual medical diagnosis.
Always consult with a healthcare professional for proper medical advice and diagnosis.
""")
