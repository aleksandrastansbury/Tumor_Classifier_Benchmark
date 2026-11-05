#Author: Eleanor Krause
#Last Edit Date: 11/5/2026
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)

def model_eval(y_test, y_pred):
    """
    A function to evaluate a model on accuracy, precision, recall, f1, 
    type I errors, type II errors, and confusion matrices. 0 is benign and 1 is malignant
    """
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "typeI(FP)": int(fp),
        "typeII(FN)": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
    }

    return results