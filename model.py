"""model.py (KNN, CART, SVM, logistic regression): builds and tunes models.
Every make_xxx() returns (pipeline, grid); tune() runs 10-fold CV on training data, scored on F1.
k_features: None = all 30 measurements, or the top k by ANOVA F-score.
Experiment loops live in run_experiments.py.
"""

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

KNN_METRICS = ["euclidean", "manhattan", "chebyshev"]

def _selector(k_features):
    """ANOVA F-score selection (keep the top k);  None = use all 30 measurements."""
    if k_features is None:
        return []
    return [("select", SelectKBest(f_classif, k=k_features))]

def make_knn(metric, k_features=None):
    """KNN uses distances, so standardize first."""
    steps = [("scale", StandardScaler())] + _selector(k_features) + \
            [("clf", KNeighborsClassifier(metric=metric))]
    grid = {"clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15]}
    return Pipeline(steps), grid

def make_cart(k_features=None):
    """Trees split on one feature at a time, so no scaling is needed.
    Unpruned: default settings grow the tree until the leaves are pure."""
    steps = _selector(k_features) + [("clf", DecisionTreeClassifier(random_state=0))]
    return Pipeline(steps), {}

#TODO : Aleks: SVM & Logistic Regression

def tune(pipe, grid, X_train, y_train, verbose=False):
    """10-fold cross-validation on the training data only, selecting the best model using F1."""
    if not grid:
        return pipe.fit(X_train, y_train)
    #search = GridSearchCV(pipe, grid, cv=10, scoring="recall")
    """End up using F1 instead: tuning on recall alone gave neighbors = 1 for every metric,
      because many settings tied on recall and the first one won. F1 also counts false alarms, so it broke the ties."""
    search = GridSearchCV(pipe, grid, cv=10, scoring="f1") 
    search.fit(X_train, y_train)
    if verbose:
        print("Best parameters:", search.best_params_)
        print("Best CV F1:", round(search.best_score_, 4))
        for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]):
            print(params, "F1:", round(score, 4))
    return search.best_estimator_

if __name__ == "__main__":
    """quick test on one split (including debugging); the full experiments live in run_experiments.py"""
    from sklearn.metrics import confusion_matrix, recall_score

    from data import get_split

    X_train, X_test, y_train, y_test = get_split(seed=0)
    models = {f"knn_{m}": make_knn(m) for m in KNN_METRICS}
    models["cart"] = make_cart()
    models["cart_top10"] = make_cart(k_features=10)
    for name, (pipe, grid) in models.items():
        model = tune(pipe, grid, X_train, y_train)
        pred = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        print(f"{name}: recall={recall_score(y_test, pred):.3f} false_neg={fn} false_pos={fp}")
        if name.startswith("knn"):
            print("   neighbors =", model.named_steps["clf"].n_neighbors)