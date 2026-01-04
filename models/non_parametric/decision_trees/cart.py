"""
Minimal CART-style Decision Trees (educational):

- Binary, axis-aligned splits
- Greedy top-down growth
- Shared base class for common logic
- Classification: Gini or Entropy, predicts class probabilities
- Regression: MSE/variance, predicts leaf mean

Design goal: readable, scikit-learn-like behavior for teaching.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class _BaseDecisionTree:
    """
    Common CART machinery shared by classifier and regressor.

    Subclasses must implement:
      - _node_value(y): leaf prediction value
      - _impurity(y): impurity for a node
      - _is_pure(y): whether node is pure (optional override)
    """

    @dataclass(slots=True)
    class _Node:
        # structure
        is_leaf: bool
        feature: int | None = None
        threshold: float | None = None
        left: "_BaseDecisionTree._Node | None" = None
        right: "_BaseDecisionTree._Node | None" = None
        # stats
        n: int = 0
        impurity: float = 0.0
        # leaf prediction (type depends on task: ndarray for clf, float for reg)
        value: object | None = None

    def __init__(
        self,
        *,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    # -------------------------
    # Hooks for subclasses
    # -------------------------
    def _node_value(self, y: np.ndarray):
        raise NotImplementedError

    def _impurity(self, y: np.ndarray) -> float:
        raise NotImplementedError

    def _is_pure(self, y: np.ndarray) -> bool:
        # Default: "all y equal"
        return bool(np.all(y == y[0])) if y.size > 0 else True

    # -------------------------
    # Best split search
    # -------------------------
    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[int | None, float | None, float, float]:
        """
        Return (best_feature, best_threshold, best_gain, parent_impurity).

        Gain = I(parent) - wL*I(left) - wR*I(right)
        """
        n, d = X.shape
        parent_impurity = float(self._impurity(y))

        best_gain = 0.0
        best_f: int | None = None
        best_thr: float | None = None

        for f in range(d):
            x = X[:, f]
            uniq = np.unique(x)
            if uniq.size <= 1:
                continue

            # CART uses midpoints between sorted unique values as candidate thresholds
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0

            for thr in thresholds:
                left = x <= thr
                nL = int(left.sum())
                nR = n - nL

                if nL < self.min_samples_leaf or nR < self.min_samples_leaf:
                    continue

                imp_left = self._impurity(y[left])
                imp_right = self._impurity(y[~left])

                gain = parent_impurity - (nL / n) * imp_left - (nR / n) * imp_right

                if gain > best_gain:
                    best_gain = float(gain)
                    best_f = int(f)
                    best_thr = float(thr)

        return best_f, best_thr, best_gain, parent_impurity

    # -------------------------
    # Tree growth (recursive)
    # -------------------------
    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        node = self._Node(
            is_leaf=False,
            n=int(y.size),
            impurity=float(self._impurity(y)),
        )

        # stopping conditions
        if (
            y.size < self.min_samples_split
            or self._is_pure(y)
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            node.is_leaf = True
            node.value = self._node_value(y)
            return node

        f, thr, gain, _ = self._best_split(X, y)

        if f is None or gain <= 0.0:
            node.is_leaf = True
            node.value = self._node_value(y)
            return node

        left_mask = X[:, f] <= thr

        node.feature = f
        node.threshold = thr
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    # -------------------------
    # Traversal utilities
    # -------------------------
    def _apply_one(self, x: np.ndarray, node: _Node) -> object:
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def _check_fitted(self) -> None:
        if not hasattr(self, "tree_"):
            raise ValueError("This tree is not fitted yet. Call fit(X, y) first.")


class DecisionTreeClassifier(_BaseDecisionTree):
    """
    CART-style Decision Tree Classifier (binary splits).

    criterion:
      - "gini"
      - "entropy"
    """

    def __init__(
        self,
        criterion: str = "gini",
        *,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> None:
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion

    # ---- classification-specific helpers ----
    def _counts(self, y: np.ndarray) -> np.ndarray:
        return np.bincount(y, minlength=self.n_classes_).astype(float)

    def _leaf_proba(self, y: np.ndarray) -> np.ndarray:
        counts = self._counts(y)
        s = counts.sum()
        return counts / s if s > 0 else np.full(self.n_classes_, 1.0 / self.n_classes_)

    def _gini(self, y: np.ndarray) -> float:
        p = self._leaf_proba(y)
        return 1.0 - float(np.sum(p * p))

    def _entropy(self, y: np.ndarray) -> float:
        p = self._leaf_proba(y)
        p = p[p > 0]
        return -float(np.sum(p * np.log(p)))

    # ---- hooks for base class ----
    def _node_value(self, y: np.ndarray) -> np.ndarray:
        # store full probability vector at leaves
        return self._leaf_proba(y)

    def _impurity(self, y: np.ndarray) -> float:
        return self._gini(y) if self.criterion == "gini" else self._entropy(y)

    def _is_pure(self, y: np.ndarray) -> bool:
        # faster purity check in encoded space: only one nonzero count
        if y.size == 0:
            return True
        c = self._counts(y)
        return int(np.count_nonzero(c)) <= 1

    # ---- public API ----
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X = np.asarray(X, float)

        # encode labels to 0..K-1 for bincount
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        self.tree_ = self._build(X, y_enc, depth=0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, float)
        return np.vstack([self._apply_one(x, self.tree_) for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class DecisionTreeRegressor(_BaseDecisionTree):
    """
    CART-style Decision Tree Regressor (binary splits).

    criterion:
      - "mse" (a.k.a. variance reduction / squared error reduction)
    """

    def __init__(
        self,
        criterion: str = "mse",
        *,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> None:
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        if criterion != "mse":
            raise ValueError("criterion must be 'mse' for this educational regressor")
        self.criterion = criterion

    # ---- regression impurity and leaf prediction ----
    def _node_value(self, y: np.ndarray) -> float:
        # leaf prediction is mean
        return float(np.mean(y)) if y.size > 0 else 0.0

    def _impurity(self, y: np.ndarray) -> float:
        # impurity = MSE around the mean = variance (up to normalization)
        if y.size == 0:
            return 0.0
        mu = float(np.mean(y))
        return float(np.mean((y - mu) ** 2))

    # ---- public API ----
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        y = y.reshape(-1)  # continuous targets
        self.tree_ = self._build(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.array([self._apply_one(x, self.tree_) for x in X], dtype=float)
