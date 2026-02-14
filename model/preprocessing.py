import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def prepare_data(
    df,
    target_col=None,
    test_size=0.2,
    scale_features=True,
    remove_duplicates=True,
    drop_missing=True
):
    # 1) Identify target column
    if target_col is None:
        target_col = df.columns[-1]

    # 2) Basic cleaning
    df_clean = df.copy()

    if remove_duplicates:
        df_clean = df_clean.drop_duplicates()

    if drop_missing:
        df_clean = df_clean.dropna()

    # 3) Split features + target
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    # 4) Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 5) Train test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=42,
        stratify=y_encoded
    )

    # 6) Scaling (only for LR/KNN/SVM)
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.values
        X_test = X_test.values

    return X_train, X_test, y_train, y_test, scaler, le, target_col, df_clean
