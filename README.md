# Emotion Detection using Machine Learning

Currently, this project only takes in images which then gets predicted by the model, we are working on the live webcam feed. [ issue:#1 ]

This project implements an emotion detection system using machine learning and computer vision techniques. The system captures real-time video through a webcam, processes the input frames, and predicts the emotion displayed on the user's face.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model](#model)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

## Overview

This project is designed to detect emotions such as happy, sad, angry, surprised, and neutral by analyzing facial expressions in real-time. It uses a convolutional neural network (CNN) trained on a facial expression dataset to classify emotions and integrates OpenCV for real-time webcam input.

## Dataset

The model is trained on the [FER-2013 dataset](https://www.kaggle.com/msambare/fer2013), which contains images of faces labeled with 7 different emotions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Model

The emotion detection model is built using a Convolutional Neural Network (CNN) architecture. The model has been trained using TensorFlow and Keras to recognize emotions from facial images.

- **Input**: 48x48 grayscale image of a face
- **Output**: Probability distribution across 7 emotion classes

## Features

- Real-time emotion detection using a webcam.
- Integration with OpenCV to capture live video.
- Predicts 7 different emotions based on facial expressions.
- User-friendly interface with `streamlit` for easy interaction.
  
## Installation

### Prerequisites

Before running this project, ensure you have the following installed:

- Python 3.7 or higher
- pip
- OpenCV
- TensorFlow/Keras

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/emotion-detection.git
   cd emotion-detection

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. (Optional) Install system dependencies for OpenCV (if needed):
   ```bash
   sudo apt-get install libgl1-mesa-glx
   sudo apt-get install libglib2.0-0

Once everything is installed, you can run the application locally:

1. Run the streamlit app:
   ```bash
   streamlit run app.py


## Technologies Used

- Python: Programming language used to implement the project.
- TensorFlow & Keras: For building and training the neural network model.
- OpenCV: For real-time video processing.
- Streamlit: For creating the user interface and handling live webcam feeds.
- NumPy & Pandas: For data manipulation and preprocessing.


## Contributing
- If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. Contributions are welcome!
