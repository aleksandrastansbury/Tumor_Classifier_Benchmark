# TODO: need to add more documentation here eventually but I want to go to bed 

import warnings
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from data import get_split, N_SEEDS
from id3 import ID3Classifier

warnings.filterwarnings("ignore") # KBinsDiscretizer warns when bins collapse

results = []
for seed in range(N_SEEDS):
    Xtr, Xte, ytr, yte = get_split(seed)
    model = ID3Classifier(n_bins=5).fit(Xtr, ytr)
    pred = model.predict(Xte)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    nodes, depth = model.size()
    results.append({
        "accuracy":  accuracy_score(yte, pred),
        "precision": precision_score(yte, pred),
        "recall":    recall_score(yte, pred), # malignant recall
        "f1":        f1_score(yte, pred),
        "type1_FP":  fp,
        "type2_FN":  fn,
        "train_acc": accuracy_score(ytr, model.predict(Xtr)),
        "nodes":     nodes,
    })
    print(f"seed {seed}: acc={results[-1]['accuracy']:.3f} "
          f"recall={results[-1]['recall']:.3f} FP={fp} FN={fn} nodes={nodes}")

print("\nMean +/- std over", N_SEEDS, "seeds")
for k in results[0]:
    vals = np.array([r[k] for r in results], dtype=float)
    print(f"  {k:10s} {vals.mean():.3f} +/- {vals.std():.3f}")