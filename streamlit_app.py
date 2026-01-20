import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import io

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
    st.write("### Features:")
    st.write("✓ CSV Dataset Upload")
    st.write("✓ Model Selection Dropdown")
    st.write("✓ Evaluation Metrics Display")
    st.write("✓ Confusion Matrix & Classification Report")

elif page == "Model Training":
    st.header("Model Training")
    
    # Feature A: Dataset upload option (CSV)
    st.subheader("A. Upload Dataset (CSV)")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")
        st.write(f"Dataset shape: {df.shape}")
        st.write("First few rows:")
        st.dataframe(df.head())
        
        # Feature B: Model selection dropdown
        st.subheader("B. Select Model")
        model_choice = st.selectbox(
            "Select a classification model:",
            ["Logistic Regression", "Random Forest", "Support Vector Machine"]
        )
        st.write(f"Selected model: **{model_choice}**")
        
        # Prepare data for training
        st.subheader("Data Preparation")
        
        # Assuming last column is the target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the selected model
        if st.button("Train Model"):
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=200, random_state=42)
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # Support Vector Machine
                model = SVC(kernel='rbf', random_state=42)
            
            # Train the model
            with st.spinner("Training model..."):
                model.fit(X_train_scaled, y_train)
            
            st.success("Model trained successfully!")
            
            # Make predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Feature C: Display evaluation metrics
            st.subheader("C. Evaluation Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Training Metrics:**")
                st.metric("Accuracy", f"{accuracy_score(y_train, y_train_pred):.4f}")
                st.metric("Precision", f"{precision_score(y_train, y_train_pred, average='weighted', zero_division=0):.4f}")
                st.metric("Recall", f"{recall_score(y_train, y_train_pred, average='weighted', zero_division=0):.4f}")
                st.metric("F1-Score", f"{f1_score(y_train, y_train_pred, average='weighted', zero_division=0):.4f}")
            
            with col2:
                st.write("**Testing Metrics:**")
                st.metric("Accuracy", f"{accuracy_score(y_test, y_test_pred):.4f}")
                st.metric("Precision", f"{precision_score(y_test, y_test_pred, average='weighted', zero_division=0):.4f}")
                st.metric("Recall", f"{recall_score(y_test, y_test_pred, average='weighted', zero_division=0):.4f}")
                st.metric("F1-Score", f"{f1_score(y_test, y_test_pred, average='weighted', zero_division=0):.4f}")
            
            # Feature D: Confusion matrix and classification report
            st.subheader("D. Confusion Matrix & Classification Report")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.write("**Confusion Matrix (Test Set):**")
                cm = confusion_matrix(y_test, y_test_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
            
            with col4:
                st.write("**Classification Report (Test Set):**")
                report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
            
            # Store model in session state for predictions
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.feature_count = X.shape[1]
    else:
        st.info("Please upload a CSV file to start training.")

elif page == "Predictions":
    st.header("Make Predictions")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first on the 'Model Training' page.")
    else:
        st.write("Use the trained model to make predictions on new data.")
        st.info("Feature coming soon: Make predictions on new data points.")
