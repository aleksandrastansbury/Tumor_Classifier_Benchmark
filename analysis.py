"""
analysis.py : turns each results/expN.csv into tables, t-tests, and charts.
One function per experiment. Run after run_experiments.py: python analysis.py
"""

import matplotlib

matplotlib.use("Agg")                      
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_rel

METRICS = ["recall", "accuracy", "f1", "precision", "typeII(FN)", "typeI(FP)"]

def analyze_exp1(path="results/exp1.csv"):
    """TODO (Ella K): H1 """
    print("analyze_exp1: TODO")

def analyze_exp2(path="results/exp2.csv"):
    """H2: compare the three KNN distances; Chebyshev is predicted to have the lowest malignant recall."""
    df = pd.read_csv(path)

    # results table: mean and standard deviation over the 10 seeds
    print("\n Experiment 2: mean and std over seeds")
    print(df.groupby("model")[METRICS].agg(["mean", "std"]).round(3))

    # which number of neighbors each distance picked
    print("\n neighbors chosen by tuning")
    print(df.groupby("model")["n_neighbors"].describe().round(2))

    # paired t-tests: same seed = same split, so rows are paired by seed
    def get(model, col):
        return df[df.model == model].sort_values("seed")[col].to_numpy()

    pairs = [("knn_chebyshev", "knn_euclidean"),
             ("knn_chebyshev", "knn_manhattan"),
             ("knn_euclidean", "knn_manhattan")]
    print("\n paired t-tests (negative t = first distance lower) ")
    for a, b in pairs:
        for col in ["recall", "accuracy", "f1"]:
            t, p = ttest_rel(get(a, col), get(b, col))
            note = "significant (p < 0.05)" if p < 0.05 else "not significant"
            print(f"{a} vs {b} {col:9s} t={t:6.2f}  p={p:.3f}  {note}")

    print("\n average confusion matrix per distance (over seeds)")
    for model in df.model.unique():
        m = df[df.model == model][["tn", "fp", "fn", "tp"]].mean().round(1)
        print(f"{model}: benign -> [{m.tn} correct, {m.fp} flagged malignant] | "
              f"malignant -> [{m.fn} missed, {m.tp} caught]")
        
    # chart: malignant recall per distance over the seeds
    df.boxplot(column="recall", by="model")
    plt.title("Malignant recall by KNN distance (10 seeds)")
    plt.suptitle("")
    plt.ylabel("recall")
    plt.savefig("results/exp2_recall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nsaved chart: results/exp2_recall.png")

def analyze_exp3(path="results/exp3.csv"):
    """TODO (Aleks): H3 """
    print("analyze_exp3: TODO")

def analyze_exp4(path="results/exp4.csv"):
    """TODO (Ella G): cutoffs """
    print("analyze_exp4: TODO")

if __name__ == "__main__":
    analyze_exp1()
    analyze_exp2()
    analyze_exp3()
    analyze_exp4()