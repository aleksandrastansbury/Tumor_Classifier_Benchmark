"""
run_experiments.py : runs Experiments 1-4 over seeds 0-9 and saves one CSV per experiment in results/.
Exp 1 (H1) feature count: top 5, 10, 15, all 30 by ANOVA F-score 
Exp 2 (H2) KNN distances: euclidean, manhattan, chebyshev
Exp 3 (H3) unpruned trees: ID3, C4.5, CART 
Exp 4 probability cutoffs: 0.20 to 0.50 
"""
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from data import N_SEEDS, get_split
from metrics import model_eval
from model import KNN_METRICS, make_cart, make_knn, make_logreg, make_svm, tune


def score(y_true, y_pred):
    """Run model_eval and flatten the confusion matrix into tn, fp, fn, tp columns."""
    res = model_eval(y_true, y_pred)
    (tn, fp), (fn, tp) = res.pop("confusion_matrix")
    res.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return res

def save(rows, path):
    os.makedirs("results", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
        
#Experiment 1
def run_exp1(n_seeds = N_SEEDS):
    rows = []
    count_options = [5, 10, 15, 30]
    for seed in range(n_seeds):
        X_train, X_test, y_train, y_test = get_split(seed)
        for num_features in count_options:
            if num_features != 30:
                num = num_features
            else:
                num = None
            model_types = {}
            for metric in KNN_METRICS:
                model_types["KNN " + metric] = make_knn(metric, k_features=num)
            model_types["CART"] = make_cart(k_features = num)
            model_types["SVM"] = make_svm(k_features = num)
            model_types["Logistic Regression"] = make_logreg(k_features = num)
            for name in model_types:
                pipe, grid = model_types[name]
                model = tune(pipe, grid, X_train, y_train)
                predictions = model.predict(X_test)
                results = score(y_test, predictions)
                row = {"experiment": 1,
                       "model": name,
                       "setting": num_features,
                       "seed": seed}
                row.update(results)
                rows.append(row)

    save(rows, "results/exp1.csv")
    
#Experiment 2
def run_exp2(n_seeds=N_SEEDS):
    """H2: three KNN distances on the same splits, neighbors tuned the same way."""
    rows = []
    for seed in range(n_seeds):
        X_train, X_test, y_train, y_test = get_split(seed)
        for metric in KNN_METRICS:
            pipe, grid = make_knn(metric)
            model = tune(pipe, grid, X_train, y_train)
            row = {"experiment": 2, "model": f"knn_{metric}", "setting": "all30",
                   "seed": seed, "n_neighbors": model.named_steps["clf"].n_neighbors}
            row.update(score(y_test, model.predict(X_test)))
            rows.append(row)
    save(rows, "results/exp2.csv")

# TODO: Aleks; Experiment 3 (unpruned trees, H3)

# Experiment 4 
CUTOFFS = [0.20, 0.30, 0.40, 0.50] 
CUTOFF_BETA = 2  # F2 weights recall 2x as much as precision (a missed tumor have a larger affect compared to a false alarm)

def pick_cutoff(model, X_train, y_train, seed):
    """Choose a cutoff using 10-fold CV on the training data only (test set never touched).
    Cross_val_predict refits a clone of the tuned pipeline on 9 folds and predicts the 10th,
    so every training tumor gets an out-of-fold probability. We score each cutoff on those
    and keep the best F2. Ties go to the higher cutoff (fewer false alarms)."""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed) # splitting training data into 10 groups/folds
    p_oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    best_cutoff, best_f = None, -1.0  # trying each cutoff and tracking the best so far
    for c in sorted(CUTOFFS, reverse=True):  # sorting from highest cutoff to lowest
        f = fbeta_score(y_train, (p_oof >= c).astype(int), beta=CUTOFF_BETA, zero_division=0)
        if f > best_f:
            best_cutoff, best_f = c, f
    return best_cutoff, best_f # returning best cutoff and its F2 score on the training data

def run_exp4(n_seeds=N_SEEDS):
    """Tune each model once, then turn its probabilities into labels at every cutoff.
    All cutoffs are reported on the test set; `selected` = 1 marks the one CV chose."""
    rows = [] 
    for seed in range(n_seeds): 
        X_train, X_test, y_train, y_test = get_split(seed)

        # comparing the 5 models: each make_ returns (pipline, grid): pipeline is the model; the grid lists settings to try during tuning.
        models = {f"knn_{m}": make_knn(m) for m in KNN_METRICS}
        models["svm"] = make_svm()
        models["logreg"] = make_logreg()

        for name, (pipe, grid) in models.items():
            # finding the best model setting using training data. done once, cutoffs reuse the same fitted model. 
            model = tune(pipe, grid, X_train, y_train)      # hyperparameters: training data only
            chosen, cv_f2 = pick_cutoff(model, X_train, y_train, seed) # choose a cutoff using only training data 
           # get each test tumor's probability of being malignant. 
            p_test = model.predict_proba(X_test)[:, 1]      # P(malignant); label 1 = malignant

            for c in CUTOFFS: # apply every cutoff to same probabilities and score it. 
                # true (malignant) when probability >= cutoff, then convert to 1/0
                y_pred = (p_test >= c).astype(int)
                row = {"experiment": 4, "model": name, "setting": f"cutoff{c:.2f}",
                       "seed": seed, "cutoff": c, "selected": int(c == chosen), # selected = 1 only on the row whose cutoff CV picked 
                       "chosen_cutoff_cv_f2": round(cv_f2, 4)} # training-data F2 score of the chosen cutoff 
                row.update(score(y_test, y_pred)) # adds accuracy, precision, F1, False positives, false negatives, recall, and confusion matrix counts to this row. 
                rows.append(row)
    # 5 models * 4 cutoffs * 10 seeds = 200 rows written to the CSV
    save(rows, "results/exp4.csv")

EXPERIMENTS = {
    "1": run_exp1,   
    "2": run_exp2,   
#   "3": run_exp3,   
    "4": run_exp4,
}

if __name__ == "__main__":
    for w in (sys.argv[1:] or list(EXPERIMENTS)):
        print("running experiment", w)
        EXPERIMENTS[w]()