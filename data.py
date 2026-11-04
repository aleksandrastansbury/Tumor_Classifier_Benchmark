import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

N_SEEDS = 10  

def load_data():
    d = load_breast_cancer()
    X = d.data
    y = 1 - d.target
    return X, y, list(d.feature_names)

def get_split(seed):
    X, y, _ = load_data()
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

if __name__ == "__main__":
    d = load_breast_cancer()
    print(d.target_names)           
    print(np.bincount(d.target))     # [212 357] ; original: label 0 has 212
    X, y, names = load_data()
    print(np.bincount(y))            # [357 212] ; after flip: label 1 has 212
    print(X.shape, "malignant:", int(y.sum()), "benign:", int((1 - y).sum()))

# Why I flip the labels (Uncomment if you wanna test how flip works)
"""
from sklearn.metrics import recall_score
y_true = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]   # original labels: 0 = malignant (3 tumors)
y_pred = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]   # the model missed one malignant tumor
print(recall_score(y_true, y_pred))        # 1.0 (perfect score we got even though it is missing to catch one malignant as this measures benign (label 1))
flip = lambda labels: [1 - v for v in labels]
print(recall_score(flip(y_true), flip(y_pred)))   # when flip 0.67 : correct malignant recall (2 of 3 caught)
"""