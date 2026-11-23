"""
metrics.py: Provides functions to evaluate classification model performance.
"""
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def model_eval(y_test, y_pred, metrics=None):
    """
    A function to evaluate a model on accuracy, precision, recall, f1, 
    type I errors, type II errors, and confusion matrices. 0 is benign and 1 is malignant
    """
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()

    if metrics is None:
        metrics = {"accuracy", "precision", "recall", "f1", "typeI(FP)", "typeII(FN)", "confusion_matrix"}
    
    results = {}

    if "accuracy" in metrics:
        results["accuracy"] = accuracy_score(y_test, y_pred)

    if "precision" in metrics:
        results["precision"] = precision_score(y_test, y_pred)

    if "recall" in metrics:
        results["recall"] = recall_score(y_test, y_pred)

    if "f1" in metrics:
        results["f1"] = f1_score(y_test, y_pred)

    if "typeI(FP)" in metrics:
        results["typeI(FP)"] = int(fp)

    if "typeII(FN)" in metrics:
        results["typeII(FN)"] = int(fn)

    if "confusion_matrix" in metrics:
        results["confusion_matrix"] = [[int(tn), int(fp)], [int(fn), int(tp)]]

    return results

if __name__ == "__main__":
    # Quick test using sample actual and predicted labels.
    print(model_eval([1, 1, 0, 0, 1], [1, 0, 0, 1, 1]))