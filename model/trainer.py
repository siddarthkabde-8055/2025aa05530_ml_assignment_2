import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

from sklearn.preprocessing import label_binarize


# -------------------------------
# Internal helper for metrics
# -------------------------------
def _calculate_metrics_multiclass(y_test, y_pred, y_proba, classes):
    y_test_bin = label_binarize(y_test, classes=classes)

    auc_score = roc_auc_score(
        y_test_bin,
        y_proba,
        average="macro",
        multi_class="ovr"
    )

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc_score,
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return results



# -------------------------------
# 1) Logistic Regression
# -------------------------------
def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    results = _calculate_metrics_multiclass(y_test, y_pred, y_proba, model.classes_)


    return model, results, y_pred


# -------------------------------
# 2) Decision Tree
# -------------------------------
def train_decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    results = _calculate_metrics_multiclass(y_test, y_pred, y_proba, model.classes_)


    return model, results, y_pred


# -------------------------------
# 3) KNN
# -------------------------------
def train_knn(X_train, y_train, X_test, y_test, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    results = _calculate_metrics_multiclass(y_test, y_pred, y_proba, model.classes_)


    return model, results, y_pred


# -------------------------------
# 4) Naive Bayes (Gaussian)
# -------------------------------
def train_naive_bayes(X_train, y_train, X_test, y_test):
    model = GaussianNB()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    results = _calculate_metrics_multiclass(y_test, y_pred, y_proba, model.classes_)


    return model, results, y_pred


# -------------------------------
# 5) Random Forest (Ensemble)
# -------------------------------
def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=200):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    results = _calculate_metrics_multiclass(y_test, y_pred, y_proba, model.classes_)


    return model, results, y_pred


# -------------------------------
# 6) XGBoost (Ensemble)
# -------------------------------
def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Requires xgboost to be installed:
    pip install xgboost
    """

    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError(
            "XGBoost is not installed. Please run: pip install xgboost"
        )

    num_classes = len(np.unique(y_train))

    model = XGBClassifier(
        n_estimators=50,          # reduce
        learning_rate=0.1,
        max_depth=4,              # reduce
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1                  # IMPORTANT for Streamlit cloud
    )


    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    results = _calculate_metrics_multiclass(y_test, y_pred, y_proba, model.classes_)

    return model, results, y_pred
