import pandas as pd
import streamlit as st


def validate_uploaded_csv(
    df: pd.DataFrame,
    mode: str = "train",
    expected_feature_count: int = 16,
    expected_target_name: str = "Class",
    min_rows: int = 500,
    max_rows_warning: int = 5000
):
    """
    Validates uploaded CSV dataset for ML training/prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Uploaded dataframe.
    mode : str
        "train"  -> expects features + target
        "predict" -> expects only features
    expected_feature_count : int
        Dry Bean dataset has 16 numeric features.
    expected_target_name : str
        Default Dry Bean target column name = "Class".
    min_rows : int
        Assignment requires minimum 500 instances.
    max_rows_warning : int
        Free tier memory warning threshold.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """

    # -------------------------
    # 1) Basic empty check
    # -------------------------
    if df is None or df.empty:
        st.error("❌ Uploaded CSV is empty.")
        return False

    # -------------------------
    # 2) Rows validation
    # -------------------------
    if df.shape[0] < min_rows and mode == "train":
        st.error(
            f"❌ Dataset has only {df.shape[0]} rows. "
            f"Minimum required is {min_rows} rows."
        )
        return False

    if df.shape[0] > max_rows_warning:
        st.warning(
            f"⚠️ Large dataset uploaded ({df.shape[0]} rows). "
            f"Streamlit free tier may run slowly or crash."
        )

    # -------------------------
    # 3) Column count validation
    # -------------------------
    if mode == "train":
        # Must have features + target
        if df.shape[1] < expected_feature_count + 1:
            st.error(
                f"❌ Dataset must contain at least {expected_feature_count} feature columns "
                f"+ 1 target column."
            )
            return False

    if mode == "predict":
        # Must have only feature columns
        if df.shape[1] < expected_feature_count:
            st.error(
                f"❌ Prediction CSV must contain at least {expected_feature_count} feature columns."
            )
            return False

    # -------------------------
    # 4) Target column validation
    # -------------------------
    if mode == "train":
        # Prefer explicit Class column if exists
        if expected_target_name in df.columns:
            target_col = expected_target_name
        else:
            # fallback: assume last column is target
            target_col = df.columns[-1]
            st.warning(
                f"⚠️ Target column '{expected_target_name}' not found. "
                f"Using last column '{target_col}' as target."
            )

        # Target must not be numeric only
        if pd.api.types.is_numeric_dtype(df[target_col]):
            st.warning(
                f"⚠️ Target column '{target_col}' looks numeric. "
                "Make sure it contains class labels."
            )

    # -------------------------
    # 5) Feature columns validation
    # -------------------------
    if mode == "train":
        feature_cols = [c for c in df.columns if c != target_col]
    else:
        feature_cols = list(df.columns)

    # Only take first expected_feature_count columns
    feature_cols = feature_cols[:expected_feature_count]

    non_numeric = [
        c for c in feature_cols
        if not pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(non_numeric) > 0:
        st.error(
            "❌ Some feature columns are NOT numeric:\n\n"
            + ", ".join(non_numeric)
        )
        return False

    # -------------------------
    # 6) Missing values warning
    # -------------------------
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        st.warning(
            f"⚠️ Dataset contains {int(total_missing)} missing values. "
            "Your preprocessing step may drop them."
        )

    st.success("✅ CSV validation passed.")
    return True
