import streamlit as st
import av
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set layout to wide mode
st.set_page_config(layout="wide", page_title="Emotion Detection | ML Model", page_icon="😃")

# Streamlit app title
st.html("<center><h1>Facial Emotion Recognition</h1></center>")

# Create two columns
col1, col2 = st.columns([4, 2])

# Add content to the first column
with col1:
    # Load the correct pre-trained model
    model = load_model("C:/Users/hamid/Downloads/model.keras")

    # Define emotions (update based on your training data if necessary)
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    op = {i: emotion for i, emotion in enumerate(emotions)}

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


# Add content to the second column (where predictions happen)
if uploaded_file is not None:
    with col2:
        st.html("<br><h2>Prediction</h2>")
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Convert to RGB since OpenCV loads images in BGR
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the uploaded image
        st.image(image_rgb, caption='Uploaded Image', width=300)

        # Preprocess the image for prediction
        img_resized = cv2.resize(image_rgb, (224, 224))  # Resize to match model input size
        img_array = img_to_array(img_resized) / 255.0  # Normalize pixel values
        input_arr = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Debug: Print image shape and pixel values to compare with notebook
        st.write("Image shape:", img_array.shape)

        # Make emotion prediction
        pred = np.argmax(model.predict(input_arr))
    
        predicted_emotion = op[pred]

        # Display the predicted emotion
        st.write(f"The predicted emotion is: **{predicted_emotion}**")

        st.markdown("<h3 style='text-align: center;'>Sample Images from the Dataset</h3>", unsafe_allow_html=True)

        st.image("C:/Users/hamid/OneDrive/Desktop/sampleimage.png")

with col1:
    # After the prediction, display the sample images from the dataset
    st.html("<br><br><br><br>")

    st.image("C:/Users/hamid/OneDrive/Desktop/newplot.jpg")
    