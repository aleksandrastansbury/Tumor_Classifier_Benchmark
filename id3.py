# ID3 implementation for the WDBC breast cancer benchmark.
#
# Adapted from: https://github.com/vidhikhatwani/Decision-Tree-ID3-Algorithm
#
# We could not use the repo directly without adaptation because
#the repo is a single Jupyter notebook written for the 14-row "Play Tennis"
# toy dataset, and its code is tied to that dataset in ways that break on ours:
#   - It treats the label as the strings 'Yes'/'No'; our labels are 0/1
#     (1 = malignant).
#   - The information gain formula divides by a hardcoded 14 (the number of
#     tennis rows) instead of the number of rows at the current node, so the
#     gains are wrong for any other dataset size.
#   - It relies on global variables (X, a fixed 4-item attribute list) and a
#     local file path, so it cannot be imported or reused across our 10 seeds.
#   - It only handles a few categorical values per column. Our 30 measurements
#     are continuous, so every unique value would become its own branch.
#   - It has no predict function, only a tree printout, so it cannot classify
#     unseen tumors or produce the metrics (recall, F1, confusion matrix) and
#     probabilities our experiments need.
#
# We kept the core ID3 algorithm (entropy, information gain, pick the
# attribute with the highest gain, split into one branch per value, recurse),
# with no pruning, as described in our proposal.
#
# What we changed:
#   - Labels are generic 0/1 and the gain weights use the real row count.
#   - Continuous features are binned into n_bins quantile buckets. The bin
#     edges are learned from the training data only, then reused on test data,
#     so no test information leaks in.
#   - Added fit/predict/predict_proba (class proportion at the leaf) and
#     size() (node count and depth) so ID3 plugs into the same evaluation
#     code as our other models.
#   - If a test sample reaches a bin value never seen on that path in
#     training, we stop at that node and use its majority class instead of
#     crashing.

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


class _Node:
    __slots__ = ("feature", "children", "proba", "majority")

    def __init__(self, proba, majority):
        self.feature = None # column index split on (None = leaf)
        self.children = {} # bin value -> _Node
        self.proba = proba  # P(malignant) among training rows that reached this node
        self.majority = majority # majority class at this node


def _entropy(y):
    p = np.bincount(y, minlength=2) / len(y)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


class ID3Classifier:
    """ID3: information gain, multiway splits on binned features, no pruning."""

    def __init__(self, n_bins=5):
        self.n_bins = n_bins

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        self.disc_ = KBinsDiscretizer(self.n_bins, encode="ordinal", strategy="quantile")
        Xb = self.disc_.fit_transform(X).astype(int) # bin edges learned from TRAIN only
        self.root_ = self._build(Xb, y, list(range(Xb.shape[1])))
        return self

    def _build(self, Xb, y, features):
        p = y.mean()
        node = _Node(proba=p, majority=int(p >= 0.5))
        if p in (0.0, 1.0) or not features: # pure, or out of attributes
            return node
        h = _entropy(y)
        best_gain, best_f = 0.0, None
        for f in features:
            gain = h
            for v in np.unique(Xb[:, f]):
                m = Xb[:, f] == v
                gain -= m.mean() * _entropy(y[m]) # weight = |subset| / |current rows|
            if gain > best_gain:
                best_gain, best_f = gain, f
        if best_f is None: # no attribute gives positive gain
            return node
        node.feature = best_f
        rest = [f for f in features if f != best_f]
        for v in np.unique(Xb[:, best_f]):
            m = Xb[:, best_f] == v
            node.children[v] = self._build(Xb[m], y[m], rest)
        return node

    def _leaf(self, row):
        node = self.root_
        while node.feature is not None:
            nxt = node.children.get(row[node.feature])
            if nxt is None: # bin value never seen on this path in training
                break
            node = nxt
        return node

    def predict_proba(self, X):
        Xb = self.disc_.transform(X).astype(int)
        p = np.array([self._leaf(r).proba for r in Xb])
        return np.column_stack([1 - p, p])

    def predict(self, X, cutoff=0.5):
        return (self.predict_proba(X)[:, 1] >= cutoff).astype(int)

    def size(self):
        def walk(n):
            if n.feature is None:
                return 1, 0
            res = [walk(c) for c in n.children.values()]
            return 1 + sum(r[0] for r in res), 1 + max(r[1] for r in res)
        return walk(self.root_) # (node_count, depth)