"""Symmetric hyperparameter selection for every model.

All models get the same treatment: a small grid, scored by 3-fold stratified
cross-validation **on the training split only**, then a refit on the full
training split with the best setting. The test split is never touched during
selection, and reported training times include the tuning cost.

* Classical models use scikit-learn's ``GridSearchCV``.
* The VQC is not a scikit-learn estimator, so :class:`TunedVQC` runs the same
  folds in a plain loop.
* :class:`TunedQSVC` computes the fidelity kernel matrix once per feature
  scale and slices it per fold, so cross-validating ``C`` costs no extra
  circuits. It is the same model as ``QSVC`` (an SVC on a
  ``FidelityQuantumKernel``) with the kernel passed in precomputed.

Feature scale: datasets are scaled to ``[0, pi]``. The quantum models may
shrink that to ``[0, 1]``, which matters for the ZZ feature map because it
encodes products like ``(pi - x_i)(pi - x_j)`` that oscillate rapidly over
the full range. The classical pipelines standardise their inputs, so the
range has no effect on them.
"""

from __future__ import annotations

import warnings

import numpy as np
from qiskit.primitives import StatevectorSampler
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.svm import SVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from .classical import make_classical
from .quantum import QuantumConfig, feature_map, make_vqc

CV_FOLDS = 3
FEATURE_SCALES = (1 / np.pi, 1.0)  # [0, 1] and [0, pi]

CLASSICAL_GRIDS = {
    "svm_rbf": {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.1, 1.0]},
    "svm_poly": {"svc__C": [0.1, 1, 10, 100], "svc__degree": [2, 3]},
    "logreg": {"logisticregression__C": [0.01, 0.1, 1, 10, 100]},
    "mlp": {"mlpclassifier__hidden_layer_sizes": [(16,), (16, 16)],
            "mlpclassifier__alpha": [1e-4, 1e-2]},
}
VQC_GRID = {"feature_scale": list(FEATURE_SCALES), "ansatz_reps": [1, 3]}
QSVC_GRID = {"feature_scale": list(FEATURE_SCALES), "C": [0.1, 1, 10, 100]}


def _folds(y: np.ndarray, seed: int):
    return list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def tuned_classical(name: str, seed: int) -> GridSearchCV:
    """Classical pipeline wrapped in a 3-fold grid search."""
    return GridSearchCV(
        make_classical(name, seed),
        CLASSICAL_GRIDS[name],
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed),
        scoring="accuracy",
    )


class TunedVQC:
    """VQC with feature scale and ansatz depth chosen by cross-validation."""

    def __init__(self, num_features: int, seed: int, config: QuantumConfig = QuantumConfig(),
                 grid: dict | None = None):
        self.num_features = num_features
        self.seed = seed
        self.config = config
        self.grid = grid or VQC_GRID

    def _build(self, params: dict):
        cfg = QuantumConfig(
            feature_map_reps=self.config.feature_map_reps,
            ansatz_reps=params["ansatz_reps"],
            maxiter=self.config.maxiter,
            shots=self.config.shots,
        )
        return make_vqc(self.num_features, self.seed, cfg)

    def fit(self, X: np.ndarray, y: np.ndarray) -> TunedVQC:
        self.cv_results_ = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for params in ParameterGrid(self.grid):
                scores = []
                for train, val in _folds(y, self.seed):
                    model = self._build(params)
                    Xs = X * params["feature_scale"]
                    model.fit(Xs[train], y[train])
                    scores.append(model.score(Xs[val], y[val]))
                self.cv_results_.append({**params, "cv_accuracy": float(np.mean(scores))})
            best = max(self.cv_results_, key=lambda r: r["cv_accuracy"])
            self.best_params_ = {k: best[k] for k in self.grid}
            self.model_ = self._build(self.best_params_)
            self.model_.fit(X * self.best_params_["feature_scale"], y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return np.asarray(self.model_.predict(X * self.best_params_["feature_scale"])).ravel()


class TunedQSVC:
    """SVC on a sampled fidelity kernel, with feature scale and ``C`` chosen by CV."""

    def __init__(self, num_features: int, seed: int, config: QuantumConfig = QuantumConfig(),
                 grid: dict | None = None):
        self.num_features = num_features
        self.seed = seed
        self.config = config
        self.grid = grid or QSVC_GRID

    def _kernel(self) -> FidelityQuantumKernel:
        sampler = StatevectorSampler(default_shots=self.config.shots, seed=self.seed)
        return FidelityQuantumKernel(
            feature_map=feature_map(self.num_features, self.config.feature_map_reps),
            fidelity=ComputeUncompute(sampler=sampler),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> TunedQSVC:
        kernel = self._kernel()
        gram = {s: kernel.evaluate(X * s) for s in self.grid["feature_scale"]}
        folds = _folds(y, self.seed)
        self.cv_results_ = []
        for params in ParameterGrid(self.grid):
            K = gram[params["feature_scale"]]
            scores = [
                SVC(kernel="precomputed", C=params["C"])
                .fit(K[np.ix_(tr, tr)], y[tr])
                .score(K[np.ix_(va, tr)], y[va])
                for tr, va in folds
            ]
            self.cv_results_.append({**params, "cv_accuracy": float(np.mean(scores))})
        best = max(self.cv_results_, key=lambda r: r["cv_accuracy"])
        self.best_params_ = {k: best[k] for k in self.grid}
        self.kernel_ = kernel
        self.X_train_ = X * self.best_params_["feature_scale"]
        self.svc_ = SVC(kernel="precomputed", C=self.best_params_["C"]).fit(
            gram[self.best_params_["feature_scale"]], y
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        K = self.kernel_.evaluate(X * self.best_params_["feature_scale"], self.X_train_)
        return self.svc_.predict(K)


def make_tuned(name: str, num_features: int, seed: int, config: QuantumConfig = QuantumConfig()):
    """Tuned version of any benchmark model."""
    if name in CLASSICAL_GRIDS:
        return tuned_classical(name, seed)
    if name == "vqc":
        return TunedVQC(num_features, seed, config)
    if name == "qsvc":
        return TunedQSVC(num_features, seed, config)
    raise ValueError(f"unknown model {name!r}")


def best_params(model) -> dict:
    """Selected hyperparameters, with feature scales shown as a readable range."""
    params = dict(model.best_params_)
    if "feature_scale" in params:
        params["feature_scale"] = "[0, 1]" if params["feature_scale"] < 1 else "[0, pi]"
    return {k.split("__")[-1]: v for k, v in params.items()}
