"""
ml/anomaly_detector.py
-----------------------
Isolation Forest anomaly detector — pure Python, no external dependencies.

What is Isolation Forest?
  An anomaly detection algorithm that works by randomly partitioning features.
  Normal points require many splits to isolate (they blend in with neighbors).
  Anomalies require very few splits (they're already isolated/extreme).

  Intuition: If you pick a random feature and a random split point,
  an anomaly ends up alone very quickly. A normal point takes many tries
  to separate from its neighbors.

  Score = average path length across all trees, normalized:
    - Short path → anomaly   → score close to 1.0
    - Long path  → normal    → score close to 0.0
    - Score > 0.6 → anomaly threshold (tunable)

Why this is better than threshold rules:
  Rules like "flag if > 10 MB" miss context. Isolation Forest
  considers ALL 45 features simultaneously and flags flows that
  are unusual in any combination of dimensions — catching novel
  attack patterns that no fixed rule would catch.

Paper: Liu, Ting, Zhou (2008) — "Isolation Forest"
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import json
import os

from core.feature_extractor import FeatureVector, FEATURE_NAMES, NUM_FEATURES


# ── Isolation Tree Node ───────────────────────────────────────────────────────

class IsolationNode:
    """
    A single node in an Isolation Tree.
    Either a split node (has children) or a leaf (terminal).
    """
    __slots__ = ["feature_idx", "split_value", "left", "right", "size"]

    def __init__(self):
        self.feature_idx: int   = 0
        self.split_value: float = 0.0
        self.left:  Optional["IsolationNode"] = None
        self.right: Optional[IsolationNode]   = None
        self.size:  int = 0     # Number of samples at this leaf


# ── Isolation Tree ────────────────────────────────────────────────────────────

class IsolationTree:
    """
    A single tree in the Isolation Forest.
    Built by recursively picking random features and random split points.
    """

    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth
        self.root: Optional[IsolationNode] = None

    def fit(self, data: List[List[float]], rng: random.Random) -> None:
        """Build the tree from training data (list of feature vectors)."""
        self.root = self._build(data, depth=0, rng=rng)

    def path_length(self, sample: List[float]) -> float:
        """
        Traverse the tree for one sample.
        Returns the path length (depth at which it was isolated).
        """
        return self._traverse(self.root, sample, depth=0)

    def _build(self, data: List[List[float]], depth: int,
               rng: random.Random) -> IsolationNode:
        node = IsolationNode()

        # Leaf conditions: too deep, or only one unique sample
        if depth >= self.max_depth or len(data) <= 1:
            node.size = len(data)
            return node

        # Pick a random feature that has some variance
        attempts = 0
        while attempts < NUM_FEATURES:
            feat_idx = rng.randint(0, NUM_FEATURES - 1)
            values   = [row[feat_idx] for row in data]
            v_min, v_max = min(values), max(values)
            if v_max > v_min:
                break
            attempts += 1
        else:
            # All features have zero variance — make a leaf
            node.size = len(data)
            return node

        # Random split value between min and max
        split = rng.uniform(v_min, v_max)

        left_data  = [row for row in data if row[feat_idx] < split]
        right_data = [row for row in data if row[feat_idx] >= split]

        # Avoid empty branches
        if not left_data or not right_data:
            node.size = len(data)
            return node

        node.feature_idx  = feat_idx
        node.split_value  = split
        node.left  = self._build(left_data,  depth + 1, rng)
        node.right = self._build(right_data, depth + 1, rng)
        return node

    def _traverse(self, node: IsolationNode,
                   sample: List[float], depth: int) -> float:
        """Recursively traverse until leaf, return depth + leaf adjustment."""
        if node is None:
            return depth

        # Leaf node
        if node.left is None and node.right is None:
            # Apply expected path length correction for remaining samples
            return depth + self._c(node.size)

        if sample[node.feature_idx] < node.split_value:
            return self._traverse(node.left,  sample, depth + 1)
        else:
            return self._traverse(node.right, sample, depth + 1)

    @staticmethod
    def _c(n: int) -> float:
        """
        Expected path length of unsuccessful search in BST with n nodes.
        Used to normalize scores: c(n) ≈ 2*H(n-1) - 2*(n-1)/n
        where H(k) = harmonic number ≈ ln(k) + 0.5772
        """
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        h = math.log(n - 1) + 0.5772156649   # Euler-Mascheroni constant
        return 2 * h - 2 * (n - 1) / n


# ── Isolation Forest ──────────────────────────────────────────────────────────

class IsolationForest:
    """
    Isolation Forest anomaly detector.

    Builds N trees on random subsamples, then scores new samples
    by their average isolation path length across all trees.

    Score → 1.0 = anomaly, Score → 0.0 = normal.

    Usage:
        forest = IsolationForest(n_trees=100, subsample_size=256)
        forest.fit(feature_vectors)           # list of FeatureVector
        results = forest.predict(new_vectors) # list of (score, is_anomaly)
    """

    def __init__(self,
                 n_trees:        int   = 100,
                 subsample_size: int   = 256,
                 contamination:  float = 0.1,   # expected fraction of anomalies
                 random_seed:    int   = 42):
        self.n_trees        = n_trees
        self.subsample_size = subsample_size
        self.contamination  = contamination
        self._rng           = random.Random(random_seed)
        self._trees:  List[IsolationTree] = []
        self._c_norm: float = 0.0           # normalization constant
        self._threshold: float = 0.0        # score threshold for anomaly
        self._fitted: bool = False
        self._train_scores: List[float] = []

    def fit(self, vectors: List[FeatureVector]) -> "IsolationForest":
        """
        Train the forest on a list of FeatureVector objects.
        Assumes most training data is normal (standard assumption).
        """
        if len(vectors) < 2:
            raise ValueError("Need at least 2 samples to fit.")

        data = [fv.features for fv in vectors]
        n    = len(data)
        sub  = min(self.subsample_size, n)

        # Normalization constant for path length
        self._c_norm = IsolationTree._c(sub)
        if self._c_norm == 0:
            self._c_norm = 1.0

        # Build trees
        self._trees = []
        max_depth = int(math.ceil(math.log2(sub))) + 1

        for _ in range(self.n_trees):
            subsample = self._rng.sample(data, sub)
            tree = IsolationTree(max_depth=max_depth)
            tree.fit(subsample, self._rng)
            self._trees.append(tree)

        # Compute scores on training data to set threshold
        self._train_scores = [self._score_sample(row) for row in data]
        self._threshold    = self._compute_threshold(self._train_scores)
        self._fitted       = True
        return self

    def score(self, vector: FeatureVector) -> float:
        """
        Return anomaly score for one FeatureVector.
        Score ∈ [0, 1]. Higher = more anomalous.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self._score_sample(vector.features)

    def predict(self, vectors: List[FeatureVector]) -> List[Tuple[float, bool]]:
        """
        Score and classify a list of FeatureVectors.
        Returns list of (score, is_anomaly) tuples.
        """
        return [(s := self.score(fv), s >= self._threshold)
                for fv in vectors]

    def annotate(self, vectors: List[FeatureVector]) -> List[FeatureVector]:
        """
        Annotate FeatureVectors with anomaly scores in-place.
        Returns the same list (mutated).
        """
        for fv in vectors:
            fv.anomaly_score = self.score(fv)
            fv.is_anomaly    = fv.anomaly_score >= self._threshold
        return vectors

    def save(self, path: str) -> None:
        """Serialize model to JSON (trees are not saved — use for threshold/stats only)."""
        meta = {
            "n_trees":        self.n_trees,
            "subsample_size": self.subsample_size,
            "contamination":  self.contamination,
            "c_norm":         self._c_norm,
            "threshold":      self._threshold,
            "fitted":         self._fitted,
            "feature_names":  FEATURE_NAMES,
            "train_score_mean": sum(self._train_scores)/len(self._train_scores) if self._train_scores else 0,
            "train_score_std":  self._score_std(),
        }
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)

    def load_threshold(self, path: str) -> None:
        """Load just the threshold from a saved metadata file."""
        with open(path) as f:
            meta = json.load(f)
        self._threshold = meta["threshold"]
        self._c_norm    = meta["c_norm"]

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ── Private helpers ────────────────────────────────────────────────────────

    def _score_sample(self, row: List[float]) -> float:
        """
        Compute anomaly score for one raw feature row.
        Score = 2^(-avg_path_length / c_norm)
        Score > 0.5 → anomaly (default threshold, refined by contamination)
        """
        avg_path = sum(tree.path_length(row) for tree in self._trees) / self.n_trees
        score    = 2 ** (-avg_path / self._c_norm)
        return round(score, 5)

    def _compute_threshold(self, scores: List[float]) -> float:
        """
        Set threshold so that approximately `contamination` fraction
        of training samples are flagged as anomalies.
        """
        if not scores:
            return 0.6
        sorted_scores = sorted(scores, reverse=True)
        cutoff_idx    = max(0, int(len(sorted_scores) * self.contamination) - 1)
        return sorted_scores[cutoff_idx]

    def _score_std(self) -> float:
        if len(self._train_scores) < 2:
            return 0.0
        mean = sum(self._train_scores) / len(self._train_scores)
        var  = sum((x - mean)**2 for x in self._train_scores) / len(self._train_scores)
        return math.sqrt(var)