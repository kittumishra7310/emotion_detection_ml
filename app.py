import streamlit as st
import av
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set layout to wide mode
st.set_page_config(layout="wide")

# Create two columns
col1, col2 = st.columns(2)

# Add content to the first column
with col1:
    st.markdown("<h2 style='text-align: center;'>Working Model</h2>", unsafe_allow_html=True)
    
     #Load the trained model and define the class names
    model = load_model('emotion.h5', compile=False)
    class_labels  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
    # Define a video processor class
    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Resize and normalize the frame to match model input requirements
            resized_frame = cv2.resize(img, (32, 32))  # Adjust size based on your model's input shape
            normalized_frame = resized_frame / 255.0  # Normalize to [0, 1]
            
            # Add batch dimension
            normalized_frame = np.expand_dims(normalized_frame, axis=0)
            
            # Use the model to make predictions
            predictions = model.predict(normalized_frame)
            predicted_class = np.argmax(predictions)
            label = class_names[predicted_class]
            
            # Display the prediction label on the frame
            cv2.putText(img, f'Class: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # Set up Streamlit application
    st.title("Sentiment analysis with Webcam")

    # Start the video stream
    webrtc_streamer(
        key="real-time-cifar10",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {"width": 640, "height": 480, "frameRate": {"ideal": 15, "max": 30}},
            "audio": False,
        },
        async_processing=True,
    )


    st.write("Press 'q' in the video window to quit.")
    st.write("Dataset (images)")

# Add content to the second column
with col2:
    st.markdown("<h2 style='text-align: center;'>Code</h2>", unsafe_allow_html=True)
    
    st.markdown("Importing Libraries")
    
    st.code(
        """
        import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix , classification_report 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')
        """
    )

    st.markdown("")