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
    
    @st.cache_data
    def load_images_from_directory(directory):
        data = []
    
        for label in os.listdir(directory):
            label_path = os.path.join(directory, label)
    
            if os.path.isdir(label_path):
                for filename in os.listdir(label_path):
                    if filename.endswith((".jpg", ".png", ".jpeg")):
                        img_path = os.path.join(label_path, filename)
    
                        # Read and resize image to 48x48 pixels
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (48, 48))
    
                        # Flatten image to a 1D array
                        img_flat = img.flatten()
                        data.append([img_flat, label])
    
        return pd.DataFrame(data, columns=["pixels", "label"])
    
    # Directory paths for train and test sets
    train_dir = "archive/train"
    test_dir = "archive/test"
    
    # Load datasets
    train_df = load_images_from_directory(train_dir)
    test_df = load_images_from_directory(test_dir)
    
    # Streamlit App Layout
    st.title("Image Classification Dataset Viewer")
    st.write(f"Train DataFrame shape: {train_df.shape}")
    st.write(f"Test DataFrame shape: {test_df.shape}")
    
    # Function to display samples using Streamlit
    def display_samples_per_class(dataframe, n=2):
        sample_data = dataframe.groupby('label', group_keys=False).apply(lambda x: x.sample(n))
    
        # Create a subplot for each image
        fig, axes = plt.subplots(len(sample_data['label'].unique()), n, figsize=(10, 10))
        fig.suptitle("Sample Images from Each Class", fontsize=16)
    
        for i, (idx, row) in enumerate(sample_data.iterrows()):
            label = row['label']
            pixels = np.array(row['pixels']).reshape(48, 48)
    
            ax = axes[i // n, i % n]
            ax.imshow(pixels, cmap='gray')
            ax.set_title(label)
            ax.axis('off')
    
        plt.tight_layout()
        st.pyplot(fig)  # Display the plot using Streamlit
    
    # Select dataset to view
    dataset_option = st.selectbox("Select Dataset", ("Train", "Test"))
    
    # Display 2 samples per class based on selected dataset
    if dataset_option == "Train":
        display_samples_per_class(train_df, n=2)
    else:
        display_samples_per_class(test_df, n=2)

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
