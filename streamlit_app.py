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
    st.title("ML Assignment 2 — Classification Models")

    st.write(
        "Train and evaluate **6 classification models** on the **UCI Dry Bean dataset** "
        "or using your own uploaded CSV file."
    )

    st.divider()

    st.subheader("✅ Required Features (Assignment)")

    st.write("**A. Dataset upload option (CSV)**")
    st.write("**B. Model selection dropdown (multiple models)**")
    st.write("**C. Display of evaluation metrics**")
    st.write("**D. Confusion matrix / classification report**")

    st.divider()
    st.info("➡️ Go to **Model Training** page to start.")

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------
elif page == "Model Training":

    st.title("Model Training")

    # ==========================================================
    # A. Dataset upload option (CSV)
    # ==========================================================
    st.subheader("A. Dataset upload option (CSV)")

    # Small test data download (simple + clean)
    if os.path.exists(TEST_DATA_PATH):
        with open(TEST_DATA_PATH, "rb") as f:
            st.download_button(
                label="⬇️ Download test_data.csv (for upload testing)",
                data=f,
                file_name="test_data.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.warning(
            "test_data.csv not found in `/data` folder.\n\n"
            "Please add `data/test_data.csv` to your GitHub project."
        )

    st.divider()

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
        if st.button("📥 Load Dry Bean Dataset from UCI", use_container_width=True):
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

    # ==========================================================
    # Dataset details (minimal, not clumsy)
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
    # Training Options (simple)
    # ==========================================================
    st.subheader("Training Options")

    col1, col2 = st.columns([1, 2])

    with col1:
        test_size = st.slider("Test size", 0.1, 0.5, 0.2)

    with col2:
        remove_duplicates = st.checkbox("Remove duplicates", value=True)
        drop_missing = st.checkbox("Drop missing rows", value=True)

    # Scaling rule
    scale_features = model_choice in ["Logistic Regression", "kNN"]

    # Prepare data
    X_train, X_test, y_train, y_test, scaler, le, target_col, df_clean = prepare_data(
        df,
        test_size=test_size,
        scale_features=scale_features,
        remove_duplicates=remove_duplicates,
        drop_missing=drop_missing
    )

    st.caption(f"Cleaned dataset shape: {df_clean.shape}")

    st.divider()

    # ==========================================================
    # Train buttons
    # ==========================================================
    colT1, colT2 = st.columns([1, 1])

    with colT1:
        train_one = st.button("🚀 Train Selected Model", use_container_width=True)

    with colT2:
        train_all = st.button("📊 Train All 6 Models", use_container_width=True)

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

    # ==========================================================
    # Train all models output
    # ==========================================================
    if train_all:
        with st.spinner("Training all 6 models... Please wait..."):
            all_results = []

            for m in get_model_list():
                scale_features_loop = m in ["Logistic Regression", "kNN"]

                X_train, X_test, y_train, y_test, _, _, _, _ = prepare_data(
                    df,
                    test_size=test_size,
                    scale_features=scale_features_loop,
                    remove_duplicates=remove_duplicates,
                    drop_missing=drop_missing
                )

                _, res, _ = train_selected_model(m, X_train, y_train, X_test, y_test)
                res["Model"] = m
                all_results.append(res)

            df_results = pd.DataFrame(all_results)
            df_results = df_results[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
            df_results = df_results.round(4)

        st.success("All models trained successfully ✅")

        st.subheader("📊 Model Comparison Table")
        st.dataframe(df_results, use_container_width=True)

        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Comparison Table as CSV",
            data=csv,
            file_name="model_comparison_results.csv",
            mime="text/csv",
            use_container_width=True
        )
