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
    df = pd.read_csv(path)
    print(df.head())

    # A graph to show accuracy differences between all of the models
    accuracy = df.groupby(["model", "setting"])["accuracy"].mean()

    print(accuracy)

    for name in df["model"].unique():
        model_data = accuracy[name]
        plt.plot([5, 10, 15, 30], model_data, label=name)
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy")
    plt.title("Experiment 1: Accuracy vs. Number of Features")
    plt.xticks([5, 10, 15, 30])
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig("results/exp1_accuracy_graph.png")
    plt.close()

    # A graph to show recall differences between all of the models
    recall = df.groupby(["model", "setting"])["recall"].mean()

    for name in df["model"].unique():
        model_data = recall[name]
        plt.plot([5, 10, 15, 30], model_data, label=name)

    plt.xlabel("Number of Features")
    plt.ylabel("Average Recall")
    plt.title("Experiment 1: Recall vs. Number of Features")
    plt.xticks([5, 10, 15, 30])
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig("results/exp1_recall.png")
    plt.close()

    # table: mean and std per model and feature count
    summary = df.groupby(["model", "setting"])[METRICS].agg(["mean", "std"]).round(3)
    print(summary)
    summary.to_csv("results/exp1_summary.csv")

     # average confusion matrix counts per model and feature count
    cm = df.groupby(["model", "setting"])[["tn", "fp", "fn", "tp"]].mean().round(1)
    print("\n average confusion matrix counts per model and feature count")
    print(cm)
    cm.to_csv("results/exp1_confusion.csv")

    # all 30 features vs top 5, paired by seed (positive t = 30 features higher)
    print("\n paired t-tests, all 30 features vs top 5")
    for name in df["model"].unique():
        d = df[df.model == name]
        d30 = d[d.setting == 30].sort_values("seed")
        d5 = d[d.setting == 5].sort_values("seed")
        for col in ["accuracy", "recall"]:
            t, p = ttest_rel(d30[col].to_numpy(), d5[col].to_numpy())
            note = "significant (p < 0.05)" if p < 0.05 else "not significant"
            print(f"{name:14s} {col:9s} t={t:6.2f}  p={p:.3f}  {note}")

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

    # save the results table 
    summary = df.groupby("model")[METRICS].agg(["mean", "std"]).round(3)
    summary.to_csv("results/exp2_summary.csv")

    # chart: average number of missed malignant tumors per distance
    miss = df.groupby("model")["typeII(FN)"].agg(["mean", "std"])
    miss.plot(kind="bar", y="mean", yerr="std", legend=False, capsize=4)
    plt.ylabel("malignant tumors missed (of 42)")
    plt.title("Missed malignant tumors by KNN distance (mean ± std, 10 seeds)")
    plt.xticks(rotation=0)
    plt.savefig("results/exp2_missed.png", dpi=150, bbox_inches="tight")
    plt.close()

def analyze_exp3(path="results/exp3.csv"):
    """H3: unpruned C4.5 and CART should reach test recall and accuracy at least as high as unpruned ID3.
    Also compares tree size and the train-test gap to see which design overfits most."""
    df = pd.read_csv(path)
    order = ["ID3", "C4.5", "CART"] # fixed model order for tables, tests and charts
    tree_cols = ["nodes", "depth", "train_accuracy", "train_test_gap"]

    # results table: mean and std over the 10 seeds
    print("\n Experiment 3: mean and std over seeds")
    summary = df.groupby("model")[METRICS + tree_cols].agg(["mean", "std"]).round(3).loc[order]
    print(summary)
    summary.to_csv("results/exp3_summary.csv")

    # how many bins ID3 chose in each seed (chosen by CV on the training data)
    print("\n ID3 bins chosen by cross-validation (number of seeds)")
    print(df[df.model == "ID3"]["n_bins"].astype(int).value_counts().sort_index())

    # average confusion matrix per tree design
    cm = df.groupby("model")[["tn", "fp", "fn", "tp"]].mean().round(1).loc[order]
    print("\n average confusion matrix counts per tree design")
    print(cm)
    cm.to_csv("results/exp3_confusion.csv")

    # paired t-tests: same seed = same split, so rows are paired by seed
    def get(model, col):
        return df[df.model == model].sort_values("seed")[col].to_numpy()

    pairs = [("C4.5", "ID3"), ("CART", "ID3"), ("CART", "C4.5")]
    print("\n paired t-tests (positive t = first tree is higher, negative t = first tree is lower)")
    print(" H3 compares C4.5 vs ID3 and CART vs ID3 on accuracy and recall")
    for a, b in pairs:
        for col in ["accuracy", "recall", "f1", "nodes", "train_test_gap"]:
            diff = get(a, col).mean() - get(b, col).mean()
            t, p = ttest_rel(get(a, col), get(b, col))
            note = "significant (p < 0.05)" if p < 0.05 else "not significant"
            print(f"{a:5s} vs {b:5s} {col:15s} mean diff={diff:+7.3f}  t={t:6.2f}  p={p:.3f}  {note}")

    # chart: malignant recall and accuracy per tree design over the seeds
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, col in zip(axes, ["recall", "accuracy"]):
        ax.boxplot([get(m, col) for m in order], tick_labels=order)
        ax.set_title(f"Test {col} (10 seeds)")
        ax.set_ylabel(col)
    plt.suptitle("Experiment 3: unpruned trees")
    plt.tight_layout()
    plt.savefig("results/exp3_recall_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()

    # chart: missed malignant tumors (FN) and false alarms (FP) per tree design
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, col, title in [(axes[0], "typeII(FN)", "Missed malignant tumors (FN)"),
                           (axes[1], "typeI(FP)", "False alarms (FP)")]:
        s = df.groupby("model")[col].agg(["mean", "std"]).loc[order]
        ax.bar(order, s["mean"], yerr=s["std"], capsize=4)
        ax.set_title(title)
        ax.set_ylabel("average count per test set (mean ± std)")
    plt.tight_layout()
    plt.savefig("results/exp3_errors.png", dpi=150, bbox_inches="tight")
    plt.close()

    # chart: tree size and train-test gap (which design overfits most)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, col, title in [(axes[0], "nodes", "Tree size (nodes)"),
                           (axes[1], "train_test_gap", "Train accuracy minus test accuracy")]:
        s = df.groupby("model")[col].agg(["mean", "std"]).loc[order]
        ax.bar(order, s["mean"], yerr=s["std"], capsize=4)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig("results/exp3_size_gap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nsaved charts: results/exp3_recall_accuracy.png, exp3_errors.png, exp3_size_gap.png")

def analyze_exp4(path="results/exp4.csv"):
    """ lowering the probability cutoff should catch more malignant tumors (fewer FN)
    but raise false alarms (more FP). Cutoffs 0.20-0.50 on KNN, SVM, logistic regression."""
    df = pd.read_csv(path) 
    order = ["knn_euclidean", "knn_manhattan", "knn_chebyshev", "svm", "logreg"] # fixed model order so tables, tests, and chart legend line up

    # results table: mean and std over the 10 seeds, one row per model + cutoff
    print("\n Experiment 4: mean and std over seeds, by model and cutoff")
    print(df.groupby(["model", "cutoff"])[METRICS].agg(["mean", "std"]).round(3))

    # which cutoff cross-validation picked (count of seeds, out of 10)
    print("\n cutoff chosen by cross-validation (number of seeds)")
    chosen = df[df.selected == 1]
    print(chosen.groupby(["model", "cutoff"]).size().unstack(fill_value=0))

    # test performance when each seed uses its own CV-chosen cutoff
    print("\n test performance at the CV-chosen cutoff (mean over seeds)")
    print(chosen.groupby("model")[METRICS].mean().round(3))

    # paired t-tests: cutoff 0.20 vs 0.50 on the same seed (same split)
    def get(model, cutoff, col):
        d = df[(df.model == model) & (df.cutoff == cutoff)].sort_values("seed")
        return d[col].to_numpy()

    print("\n paired t-tests, cutoff 0.20 vs 0.50 (positive t = 0.20 is higher, negative t = 0.20 is lower)")
    for model in order:
        for col in ["recall", "typeII(FN)", "typeI(FP)"]:
            t, p = ttest_rel(get(model, 0.2, col), get(model, 0.5, col))
            note = "significant (p < 0.05)" if p < 0.05 else "not significant"
            print(f"{model:14s} {col:11s} t={t:6.2f}  p={p:.3f}  {note}")

    # 5) chart: missed tumors (FN) and false alarms (FP) as the cutoff changes
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in [(axes[0], "typeII(FN)", "Missed malignant tumors (FN)"),
                           (axes[1], "typeI(FP)", "False alarms (FP)")]:
        for model in order:
            m = df[df.model == model].groupby("cutoff")[col].mean()
            ax.plot(m.index, m.values, marker="o", label=model)
        ax.set_title(title)
        ax.set_xlabel("probability cutoff")
        ax.set_ylabel("average count per test set")
        ax.set_xticks([0.2, 0.3, 0.4, 0.5])
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/exp4_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nsaved chart: results/exp4_tradeoff.png")

if __name__ == "__main__":
    analyze_exp1()
    analyze_exp2()
    analyze_exp3()
    analyze_exp4()
