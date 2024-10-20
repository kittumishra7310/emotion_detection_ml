# import streamlit as st
# import av
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# from keras.preprocessing.image import img_to_array
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
# import matplotlib.pyplot as plt
# import pandas as pd
# import os

# # Set layout to wide mode
# st.set_page_config(layout="wide")

# # Create two columns
# col1, col2 = st.columns(2)

# # Add content to the first column
# with col1:
#     st.markdown("<h2 style='text-align: center;'>Working Model</h2>", unsafe_allow_html=True)
    
#     # Load the correct pre-trained model
#     model = load_model("C:/Users/hamid/Downloads/model.keras")

#     # Define emotions (update based on your training data if necessary)
#     emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
#     op = {i: emotion for i, emotion in enumerate(emotions)}

#     # Streamlit app title
#     st.title("Facial Emotion Recognition")

#     # Upload an image
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])



#     #TODO
    

# # Add content to the second column
# with col2:
#     if uploaded_file is not None:
#         # Convert the uploaded file to an OpenCV image
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         image = cv2.imdecode(file_bytes, 1)

#         # Convert to RGB since OpenCV loads images in BGR
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Display the uploaded image
#         # st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

#         # Preprocess the image for prediction
#         img_resized = cv2.resize(image_rgb, (224, 224))  # Resize to match model input size
#         img_array = img_to_array(img_resized) / 255.0  # Normalize pixel values
#         input_arr = np.expand_dims(img_array, axis=0)  # Add batch dimension

#         # Debug: Print image shape and pixel values to compare with notebook
#         st.write("Image shape:", img_array.shape)
#         st.write("Pixel values (first 5):", img_array.flatten()[:5])

#         # Make emotion prediction
#         pred = np.argmax(model.predict(input_arr))
#         predicted_emotion = op[pred]

#         # Display the predicted emotion
#         st.write(f"The predicted emotion is: **{predicted_emotion}**")

#         # Display the input image with the predicted emotion
#         plt.figure(figsize=(1, 1))
#         plt.imshow(img_resized)
#         plt.title(f"Predicted Emotion: {predicted_emotion}", fontsize=10)
#         plt.axis('off')  # Turn off the axes for a cleaner display
#         st.pyplot(plt)


# with col1:
#     Displaying Sample Images from the dataset
#     @st.cache_data
#     def load_images_from_directory(directory):
#         data = []
    
#         for label in os.listdir(directory):
#             label_path = os.path.join(directory, label)
    
#             if os.path.isdir(label_path):
#                 for filename in os.listdir(label_path):
#                     if filename.endswith((".jpg", ".png", ".jpeg")):
#                         img_path = os.path.join(label_path, filename)
    
#                         # Read and resize image to 48x48 pixels
#                         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                         img = cv2.resize(img, (48, 48))
    
#                         # Flatten image to a 1D array
#                         img_flat = img.flatten()
#                         data.append([img_flat, label])
    
#         return pd.DataFrame(data, columns=["pixels", "label"])
    
#     # Directory paths for train and test sets
#     train_dir = "C:/Users/hamid/OneDrive/Desktop/emotion_dataset/train"
#     test_dir = "C:/Users/hamid/OneDrive/Desktop/emotion_dataset/test"
    
#     # Load datasets
#     train_df = load_images_from_directory(train_dir)
#     test_df = load_images_from_directory(test_dir)
    
#     # Streamlit App Layout
#     st.title("Image Classification Dataset Viewer")
#     st.write(f"Train DataFrame shape: {train_df.shape}")
#     st.write(f"Test DataFrame shape: {test_df.shape}")
    
#     # Function to display samples using Streamlit
#     def display_samples_per_class(dataframe, n=2):
#         sample_data = dataframe.groupby('label', group_keys=False).apply(lambda x: x.sample(n))
    
#         # Create a subplot for each image
#         fig, axes = plt.subplots(len(sample_data['label'].unique()), n, figsize=(10, 10))
#         fig.suptitle("Sample Images from Each Class", fontsize=16)
    
#         for i, (idx, row) in enumerate(sample_data.iterrows()):
#             label = row['label']
#             pixels = np.array(row['pixels']).reshape(48, 48)
    
#             ax = axes[i // n, i % n]
#             ax.imshow(pixels, cmap='gray')
#             ax.set_title(label)
#             ax.axis('off')
    
#         plt.tight_layout()
#         st.pyplot(fig)  # Display the plot using Streamlit
    
#     # Select dataset to view
#     dataset_option = st.selectbox("Select Dataset", ("Train", "Test"))
    
#     # Display 2 samples per class based on selected dataset
#     if dataset_option == "Train":
#         display_samples_per_class(train_df, n=2)
#     else:
#         display_samples_per_class(test_df, n=2)


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
st.set_page_config(layout="wide")

# Create two columns
col1, col2 = st.columns(2)

# Add content to the first column
with col1:
    st.markdown("<h2 style='text-align: center;'>Working Model</h2>", unsafe_allow_html=True)
    
    # Load the correct pre-trained model
    model = load_model("C:/Users/hamid/Downloads/model.keras")

    # Define emotions (update based on your training data if necessary)
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    op = {i: emotion for i, emotion in enumerate(emotions)}

    # Streamlit app title
    st.title("Facial Emotion Recognition")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


# Add content to the second column (where predictions happen)
if uploaded_file is not None:
    with col2:
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

        # Display the input image with the predicted emotion
        # plt.figure(figsize=(1, 1))
        # plt.imshow(img_resized)
        # plt.title(f"Predicted Emotion: {predicted_emotion}", fontsize=10)
        # plt.axis('off')  # Turn off the axes for a cleaner display
        # st.pyplot(plt)

    # After the prediction, display the sample images from the dataset
    with col1:
        st.markdown("<h3 style='text-align: center;'>Displaying Sample Images from the Dataset</h3>", unsafe_allow_html=True)

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
        train_dir = "C:/Users/hamid/OneDrive/Desktop/emotion_dataset/train"
        test_dir = "C:/Users/hamid/OneDrive/Desktop/emotion_dataset/test"

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
