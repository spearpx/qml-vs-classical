"""Benchmark datasets with leakage-free preprocessing.

Every split is stratified and seeded. All fitted preprocessing (PCA,
standardisation, scaling) is fitted on the training split only and then
applied to the test split. Features end up in ``[0, pi]``, the natural range
for angle-encoding feature maps; classical models see exactly the same arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, make_moons
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATASETS = ("moons", "iris", "breast_cancer")
TEST_SIZE = 0.3


@dataclass(frozen=True)
class Split:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    @property
    def num_features(self) -> int:
        return self.X_train.shape[1]


def _raw(name: str, data_seed: int) -> tuple[np.ndarray, np.ndarray]:
    if name == "moons":
        # Fixed generator seed: the dataset itself is the same for every split seed.
        return make_moons(n_samples=200, noise=0.25, random_state=data_seed)
    if name == "iris":
        # Versicolor vs virginica: the two overlapping classes (setosa is trivially separable).
        X, y = load_iris(return_X_y=True)
        mask = y > 0
        return X[mask], (y[mask] == 2).astype(int)
    if name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
        # Stratified subsample of 200 keeps quantum-kernel simulation tractable
        # (the kernel matrix needs O(n^2) circuits); every model uses this subset.
        X, _, y, _ = train_test_split(X, y, train_size=200, stratify=y, random_state=data_seed)
        return X, y
    raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")


def load_split(name: str, seed: int, pca_components: int = 4, data_seed: int = 0) -> Split:
    """Stratified 70/30 split of ``name`` with preprocessing fitted on the train part.

    Args:
        name: One of :data:`DATASETS`.
        seed: Controls the train/test split. Every model sees the same split
            for a given seed.
        pca_components: Features kept for ``breast_cancer`` (one qubit each).
        data_seed: Seed for dataset generation/subsampling (fixed across splits).
    """
    X, y = _raw(name, data_seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    if name == "breast_cancer":
        standard = StandardScaler().fit(X_train)
        pca = PCA(n_components=pca_components, random_state=seed).fit(standard.transform(X_train))
        X_train = pca.transform(standard.transform(X_train))
        X_test = pca.transform(standard.transform(X_test))

    scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(X_train)
    # Test points outside the training range are clipped rather than wrapped,
    # since angles beyond [0, pi] would alias under the feature map.
    X_train = scaler.transform(X_train)
    X_test = np.clip(scaler.transform(X_test), 0, np.pi)
    return Split(name, X_train, X_test, y_train.astype(int), y_test.astype(int))
