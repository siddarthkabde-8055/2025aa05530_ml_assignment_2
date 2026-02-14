import streamlit as st
import pandas as pd
import os

from model.data_loader import load_drybean_from_uci
from model.data_health_check import run_data_health_check
from model.preprocessing import prepare_data

from model.app_helpers import (
    train_selected_model,
    get_model_list
)

from model.visuals import (
    plot_confusion_matrix,
    get_classification_report_df
)

# ---------------------------------------------------
# Page config
# ---------------------------------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")

TEST_DATA_PATH = "data/test_data.csv"

# ---------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "Model Training"])

# ---------------------------------------------------
# HOME
# ---------------------------------------------------
if page == "Home":

    st.header("ML Assignment 2 — Classification Models")

    st.caption(
        "Train and evaluate **6 classification models** on the **UCI Dry Bean dataset** "
        "or using your own uploaded CSV file."
    )

    st.markdown("### ✅ Required Features (Assignment)")

    st.markdown(
        """
        - **A. Dataset upload option (CSV)**
        - **B. Model selection dropdown (multiple models)**
        - **C. Display of evaluation metrics**
        - **D. Confusion matrix / classification report**
        """
    )

    st.info("➡️ Go to **Model Training** page to start.")


# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------
elif page == "Model Training":

    st.header("Model Training")

    # ==========================================================
    # A. Dataset upload option (CSV)
    # ==========================================================
    st.subheader("A. Dataset upload option (CSV)")

    # -----------------------------
    # Row layout (LEFT = main flow, RIGHT = helper download)
    # -----------------------------
    left_col, right_col = st.columns([3, 2])

    with right_col:
        st.markdown("#### 📥 Test CSV (Quick Download)")
        st.caption("Use this file to test the CSV upload feature.")

        if os.path.exists(TEST_DATA_PATH):
            with open(TEST_DATA_PATH, "rb") as f:
                st.download_button(
                    label="⬇️ Download test_data.csv",
                    data=f,
                    file_name="test_data.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Missing: `data/test_data.csv`")

    with left_col:
        dataset_source = st.radio(
            "Choose dataset source:",
            ["UCI Dry Bean Dataset (Recommended)", "Upload CSV"],
            horizontal=True
        )

        df = None

        # -----------------------------
        # Load dataset
        # -----------------------------
        if dataset_source == "UCI Dry Bean Dataset (Recommended)":
            if st.button("📥 Load Dry Bean Dataset from UCI", type="primary"):
                with st.spinner("Loading dataset from UCI..."):
                    df = load_drybean_from_uci()
                    st.session_state.df = df

        else:
            uploaded_file = st.file_uploader("Upload CSV file", type="csv")

            if uploaded_file is not None:
                with st.spinner("Reading uploaded CSV..."):
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df

    # Use session state only after user loads/upload
    if "df" in st.session_state:
        df = st.session_state.df

    if df is None:
        st.info("Please load a dataset to continue.")
        st.stop()

    st.success("Dataset loaded successfully ✅")

    st.divider()

    # ==========================================================
    # Dataset details
    # ==========================================================
    with st.expander("Dataset Preview + Health Check", expanded=False):
        st.write("Shape:", df.shape)
        st.dataframe(df.head(10), use_container_width=True)
        st.divider()
        run_data_health_check(df)

    st.divider()

    # ==========================================================
    # B. Model selection dropdown
    # ==========================================================
    st.subheader("B. Model selection dropdown")

    model_choice = st.selectbox(
        "Select a classification model:",
        get_model_list()
    )

    # ==========================================================
    # Prepare data (NO training options)
    # ==========================================================
    # Fixed defaults (since training options removed)
    test_size = 0.2
    remove_duplicates = True
    drop_missing = True

    scale_features = model_choice in ["Logistic Regression", "kNN"]

    X_train, X_test, y_train, y_test, scaler, le, target_col, df_clean = prepare_data(
        df,
        test_size=test_size,
        scale_features=scale_features,
        remove_duplicates=remove_duplicates,
        drop_missing=drop_missing
    )

    st.caption(f"Cleaned dataset shape: {df_clean.shape} | Test split: {int(test_size * 100)}%")

    st.divider()

    # ==========================================================
    # Train button
    # ==========================================================
    train_one = st.button("🚀 Train Selected Model", type="primary", use_container_width=True)

    # ==========================================================
    # Train selected model output
    # ==========================================================
    if train_one:
        with st.spinner(f"Training {model_choice}..."):
            model, results, y_test_pred = train_selected_model(
                model_choice, X_train, y_train, X_test, y_test
            )

        st.success(f"{model_choice} trained successfully ✅")

        # -----------------------------
        # C. Evaluation metrics
        # -----------------------------
        st.subheader("C. Display of evaluation metrics")

        m1, m2, m3 = st.columns(3)
        m4, m5, m6 = st.columns(3)

        m1.metric("Accuracy", f"{results['Accuracy']:.4f}")
        m2.metric("AUC", f"{results['AUC']:.4f}")
        m3.metric("Precision", f"{results['Precision']:.4f}")

        m4.metric("Recall", f"{results['Recall']:.4f}")
        m5.metric("F1", f"{results['F1']:.4f}")
        m6.metric("MCC", f"{results['MCC']:.4f}")

        st.divider()

        # -----------------------------
        # D. Confusion matrix / report
        # -----------------------------
        st.subheader("D. Confusion matrix / classification report")

        tab1, tab2 = st.tabs(["📌 Confusion Matrix", "📄 Classification Report"])

        with tab1:
            fig = plot_confusion_matrix(y_test, y_test_pred)
            st.pyplot(fig)

        with tab2:
            report_df = get_classification_report_df(y_test, y_test_pred)
            st.dataframe(report_df, use_container_width=True)
