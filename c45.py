# C4.5 implementation for the WDBC breast cancer benchmark.
#
# Adapted from: https://github.com/Valdecy/C4.5  (Prof. Valdecy Pereira, UFF Brazil)
#
# Why we could not use that repo directly:
#   - It crashes on current pandas (pandas 3.x): it writes strings such as ">8.726"
#     into float columns and calls float() on a Series, both of which now raise errors.
#   - It is written as one long script that reads its own CSV and runs on import.
#   - Its tree is returned as a list of text rules, and its predict function re-parses
#     those strings by splitting on spaces, so our feature names ("mean radius") break it.
#   - Every split score is computed with row-by-row pandas loops, which is very slow
#     for 30 features x ~455 rows x 10 seeds x 10 CV folds.
#   - It has no probability output (needed for the cutoff experiment) and no tree size
#     or depth (needed for the tree-design experiment).
#
# What we kept (the C4.5 algorithm as the repo implements it):
#   - Entropy-based information GAIN RATIO (gain divided by the split's intrinsic value).
#   - Binary splits on continuous features: "feature <= threshold" vs "> threshold".
#   - No pruning here (pre_pruning="none" in the repo), so the tree grows until pure.
#
# What we changed:
#   - Vectorized NumPy (sort once per feature, cumulative class counts) instead of pandas loops.
#   - Thresholds are midpoints between consecutive distinct values (the standard choice;
#     the repo uses the data values themselves).
#   - Added C4.5's average-gain rule: only splits whose gain is at least the average gain
#     of all candidate splits are eligible. Without it, gain ratio tends to favor tiny,
#     lopsided splits that isolate a few samples.
#   - fit / predict / predict_proba / size() interface, the same as our ID3Classifier.
import numpy as np


def _entropy_counts(pos, n):
    """Entropy of a node given the count of class 1 (pos) and the total (n). Arrays OK."""
    n = np.asarray(n, dtype=float)
    p = np.divide(pos, n, out=np.zeros_like(n, dtype=float), where=n > 0)
    q = 1.0 - p
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(np.where(p > 0, p * np.log2(p), 0.0) + np.where(q > 0, q * np.log2(q), 0.0))
    return h


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "proba")

    def __init__(self, proba):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.proba = proba # fraction of malignant (label 1) training rows in this node


class C45Classifier:
    """C4.5: gain ratio, binary splits on continuous features, unpruned."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.root_ = self._build(X, y)
        return self

    def _best_split(self, X, y):
        n = len(y)
        pos_total = y.sum()
        h_parent = _entropy_counts(pos_total, n)
        cands = [] # (gain, gain_ratio, feature, threshold)
        for f in range(X.shape[1]):
            order = np.argsort(X[:, f], kind="mergesort")
            xs, ys = X[order, f], y[order]
            # a split is possible only between two DIFFERENT consecutive values
            idx = np.nonzero(xs[1:] > xs[:-1])[0] # split after position idx (left = idx+1 rows)
            if len(idx) == 0:
                continue
            n_left = (idx + 1).astype(float)
            pos_left = np.cumsum(ys)[idx]
            n_right = n - n_left
            pos_right = pos_total - pos_left
            w_l, w_r = n_left / n, n_right / n
            gain = h_parent - w_l * _entropy_counts(pos_left, n_left) \
                            - w_r * _entropy_counts(pos_right, n_right)
            split_info = -(w_l * np.log2(w_l) + w_r * np.log2(w_r))
            ratio = gain / split_info
            thr = (xs[idx] + xs[idx + 1]) / 2.0
            for g, r, t in zip(gain, ratio, thr):
                cands.append((g, r, f, t))
        if not cands:
            return None
        gains = np.array([c[0] for c in cands])
        ok = gains >= gains.mean() - 1e-12 # C4.5 average-gain rule
        ok &= gains > 1e-12 # and the split must actually help
        if not ok.any():
            return None
        best = max((c for c, keep in zip(cands, ok) if keep), key=lambda c: c[1])
        return best[2], best[3]

    def _build(self, X, y):
        node = _Node(proba=y.mean())
        if y.min() == y.max(): # pure node -> leaf
            return node
        split = self._best_split(X, y)
        if split is None: # no useful split -> leaf
            return node
        f, t = split
        mask = X[:, f] <= t
        node.feature, node.threshold = f, t
        node.left = self._build(X[mask], y[mask])
        node.right = self._build(X[~mask], y[~mask])
        return node

    def _leaf(self, row):
        node = self.root_
        while node.feature is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p = np.array([self._leaf(r).proba for r in X])
        return np.column_stack([1 - p, p])

    def predict(self, X, cutoff=0.5):
        return (self.predict_proba(X)[:, 1] >= cutoff).astype(int)

    def size(self):
        def walk(n):
            if n.feature is None:
                return 1, 0
            (nl, dl), (nr, dr) = walk(n.left), walk(n.right)
            return 1 + nl + nr, 1 + max(dl, dr)
        return walk(self.root_) # (node_count, depth)