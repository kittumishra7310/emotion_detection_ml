import streamlit as st

# Set layout to wide mode
st.set_page_config(layout="wide")

# Create two columns
col1, col2 = st.columns(2)

# Add content to the first column
with col1:
    st.markdown("<h2 style='text-align: center;'>Working Model</h2>", unsafe_allow_html=True)
    st.write("webcam here")
    st.write("prediction")
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