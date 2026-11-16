"""
run_experiments.py : runs Experiments 1-4 over seeds 0-9 and saves one CSV per experiment in results/.
Exp 1 (H1) feature count: top 5, 10, 15, all 30 by ANOVA F-score 
Exp 2 (H2) KNN distances: euclidean, manhattan, chebyshev
Exp 3 (H3) unpruned trees: ID3, C4.5, CART 
Exp 4 probability cutoffs: 0.20 to 0.50 
Plz Use get_split() from data.py, models from model.py, and model_eval from metrics.py !!!!!!
"""
import csv
import os
import sys

from data import N_SEEDS, get_split
from metrics import model_eval
from model import KNN_METRICS, make_knn, tune

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
        
#TODO : Ella K; Experiment 1 (feature count, H1)

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

# TODO: Ella G; Experiment 4 (cutoffs)

EXPERIMENTS = {
#   "1": run_exp1,   
    "2": run_exp2,   
#   "3": run_exp3,   
#   "4": run_exp4,   
}

if __name__ == "__main__":
    for w in (sys.argv[1:] or list(EXPERIMENTS)):
        print("running experiment", w)
        EXPERIMENTS[w]()
