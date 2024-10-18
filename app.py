import streamlit as st

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

import streamlit as st

# Create two columns
col1, col2, col3 = st.columns([4, 1, 4])

# Add content to the first column
with col1:
    st.markdown("<h2 style='text-align: center;'>Working Model</h2>", unsafe_allow_html=True)
    st.write("webcam here")

# Adding a divider between two columns
with col2:
    st.markdown(
        "<div style='height: 100%; width: 1px; background-color: black; margin: auto;'></div>",
        unsafe_allow_html=True
    )


# Add content to the third column
with col3:
    st.markdown("<h2 style='text-align: center;'>Code</h2>", unsafe_allow_html=True)
    st.write("This is the content in the second column.")