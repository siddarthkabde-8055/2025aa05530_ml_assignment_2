from model.trainer import (
    train_logistic_regression,
    train_decision_tree,
    train_knn,
    train_naive_bayes,
    train_random_forest,
    train_xgboost
)


def train_selected_model(model_name, X_train, y_train, X_test, y_test):
    """
    Trains one selected model and returns:
    model, results(dict), y_test_pred
    """

    if model_name == "Logistic Regression":
        return train_logistic_regression(X_train, y_train, X_test, y_test)

    if model_name == "Decision Tree":
        return train_decision_tree(X_train, y_train, X_test, y_test)

    if model_name == "kNN":
        return train_knn(X_train, y_train, X_test, y_test, n_neighbors=5)

    if model_name == "Naive Bayes":
        return train_naive_bayes(X_train, y_train, X_test, y_test)

    if model_name == "Random Forest (Ensemble)":
        return train_random_forest(X_train, y_train, X_test, y_test, n_estimators=200)

    if model_name == "XGBoost (Ensemble)":
        return train_xgboost(X_train, y_train, X_test, y_test)

    raise ValueError("Unknown model selected")


def get_model_list():
    return [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest (Ensemble)",
        "XGBoost (Ensemble)"
    ]
