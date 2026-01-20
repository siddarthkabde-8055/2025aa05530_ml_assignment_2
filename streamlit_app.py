import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="ML Assignment 2", layout="wide")

# Title and description
st.title("Machine Learning Model")
st.write("Welcome to the ML Assignment 2 application!")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "Model Training", "Predictions"])

if page == "Home":
    st.header("Welcome")
    st.write("This is a machine learning application built with Streamlit.")
    st.write("Use the sidebar to navigate through different sections.")

elif page == "Model Training":
    st.header("Model Training")
    st.write("Upload your dataset and train the model here.")
    # Add your training code here

elif page == "Predictions":
    st.header("Make Predictions")
    st.write("Use the trained model to make predictions.")
    # Add your prediction code here
