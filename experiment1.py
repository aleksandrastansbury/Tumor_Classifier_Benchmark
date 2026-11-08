#Author: Eleanor Krause
#Last Edit Date: 11/8/2026

import numpy as np
from sklearn.feature_selection import f_classif

def rank_features(X_train, y_train):
    """
    A function to rank features using ANOVA F-scores.
    High scores mean stronger individual relationships with the target classification.
    """

    f_scores, p_values = f_classif(X_train, y_train)
    rankings = np.argsort(f_scores)
    rankings = list(reversed(rankings))
    return rankings

def select_features(x_train, x_test, rankings, num_features):
    """
    A function that selects the top ranked features.
    """
    features = rankings[:num_features]
    x_train_features = x_train[:, features]
    x_test_features = x_test[:, features]
    return x_train_features, x_test_features
