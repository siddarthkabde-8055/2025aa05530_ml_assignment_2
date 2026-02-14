import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix(y_test, y_pred):
    """
    Returns a matplotlib figure for confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    return fig


def get_classification_report_df(y_test, y_pred):
    """
    Returns a pandas dataframe for classification report.
    """
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    return pd.DataFrame(report).transpose()
