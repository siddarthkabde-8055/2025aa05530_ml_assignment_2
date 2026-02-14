import pandas as pd
import numpy as np
import streamlit as st

def run_data_health_check(df: pd.DataFrame):
    st.subheader("🩺 Dataset Health Check")

    # 1) Basic info
    st.write("✅ Dataset Loaded")
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())

    # 2) Dtypes
    with st.expander("🔍 Column Data Types"):
        st.dataframe(df.dtypes.astype(str), use_container_width=True)

    # 3) Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()

    with st.expander("🧼 Missing Values Check"):
        st.write(f"Total missing values: **{int(total_missing)}**")
        st.dataframe(missing[missing > 0], use_container_width=True)

        if total_missing == 0:
            st.success("No missing values found ✅")

    # 4) Duplicates
    dup_count = df.duplicated().sum()
    with st.expander("📌 Duplicate Rows Check"):
        st.write("Duplicate rows:", int(dup_count))
        if dup_count == 0:
            st.success("No duplicate rows found ✅")

    # 5) Target column guess
    target_col = df.columns[-1]
    st.write("🎯 Target column (assumed last column):", f"**{target_col}**")

    # 6) Target distribution
    with st.expander("📊 Target Class Distribution"):
        try:
            vc = df[target_col].value_counts()
            st.dataframe(vc, use_container_width=True)
            st.write("Number of classes:", int(vc.shape[0]))
        except Exception as e:
            st.error("Could not compute class distribution.")
            st.exception(e)

    # 7) Numeric feature check
    with st.expander("🧠 Feature Validation (Numeric Check)"):
        feature_cols = df.columns[:-1]
        non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

        st.write("Feature count:", len(feature_cols))
        st.write("Non-numeric feature columns:", non_numeric)

        if len(non_numeric) == 0:
            st.success("All feature columns are numeric ✅")
        else:
            st.warning("Some feature columns are NOT numeric ❌")

    # 8) Feature statistics
    with st.expander("📈 Feature Summary (Describe)"):
        try:
            st.dataframe(df.describe(include=[np.number]).T, use_container_width=True)
        except Exception as e:
            st.error("Could not generate describe() for numeric columns.")
            st.exception(e)

    # Final summary
    st.subheader("✅ Final Summary")
    if total_missing == 0 and dup_count == 0:
        st.success("Dataset looks clean and ready for preprocessing + model training 🚀")
    else:
        st.warning("Dataset has issues (missing/duplicates). You may need cleaning before training.")
