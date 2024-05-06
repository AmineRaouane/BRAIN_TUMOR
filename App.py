import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model #type:ignore
from joblib import load
import xgboost as xgb
import pandas as pd

if 'Xception_model' not in st.session_state:
    st.session_state['Xception_model'] = load_model("./Xception_model.h5")

if 'XGB_model' not in st.session_state:
    st.session_state['XGB_model'] = load("./XGB.pkl")

st.title("Brain Tumor detector")

def preprocess_image(image):
    # Read the image file
    img_array = np.array(bytearray(image.read()), dtype=np.uint8)
    # Decode the image
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return cv2.resize(img, (224, 224))

user_input = st.file_uploader("Choose an image...", type="jpg")

with st.popover("More Info"):
    # Create two columns
    col1, col2, col3= st.columns(3)
        
    with col1:
        Mean = st.number_input("Mean", step=0.01)
        Variance = st.number_input("Variance", step=0.01)
        Entropy = st.number_input("Entropy", step=0.01)
        Skewness = st.number_input("Skewness", step=0.01)
    with col2:
        Kurtosis = st.number_input("Kurtosis", step=0.01)
        Contrast = st.number_input("Contrast", step=0.01)
        Energy = st.number_input("Energy", step=0.01)
        ASM = st.number_input("ASM", step=0.01)
    with col3:
        Homogeneity = st.number_input("Homogeneity", step=0.01)
        Dissimilarity = st.number_input("Dissimilarity", step=0.01)
        Correlation = st.number_input("Correlation", step=0.01)
        Coarseness = st.number_input("Coarseness", step=0.01)


if st.button("Predict"):
    if user_input :
        resized_image = preprocess_image(user_input)
        model = st.session_state['Xception_model']
        prediction = model.predict(np.expand_dims(resized_image, axis=0))
        Mapping = ['Glioma', 'Meningioma', 'Normal', 'Pituitary']
        st.write(f"The image is a {Mapping[np.argmax(prediction)]} image")
    else :
        st.write("No image Input")
    
    L= [Mean,Variance,Entropy,Skewness,Kurtosis,Contrast,Energy ,ASM ,Homogeneity,Dissimilarity,Correlation,Coarseness]

    if all(L) :
        Df = pd.DataFrame([[Mean, Variance,Variance**0.5, Entropy, Skewness, Kurtosis, Contrast, Energy, ASM, Homogeneity, Dissimilarity, Correlation, Coarseness]],columns= ["Mean", "Variance", "Standard Deviation", "Entropy", "Skewness", "Kurtosis", "Contrast", "Energy", "ASM", "Homogeneity", "Dissimilarity", "Correlation", "Coarseness"])
        model = st.session_state['XGB_model']
        decision = ["Normal","Tumor"]
        pred = decision[model.predict(Df)[0]]
        st.write(f"The given information is for a {pred} case")
    else :
        st.write("Some Info is missing")