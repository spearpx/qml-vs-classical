"""Classical baselines with standard, untuned hyperparameters.

The quantum models are not tuned either (fixed circuit depth, fixed optimizer
budget), so neither side gets a hyperparameter search. Each model standardises
its inputs internally; that is part of the usual pipeline for these models and
does not change the information available.
"""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSICAL_MODELS = ("svm_rbf", "svm_poly", "logreg", "mlp")


def make_classical(name: str, seed: int = 0) -> ClassifierMixin:
    """Build an unfitted classical classifier by name."""
    if name == "svm_rbf":
        model = SVC(kernel="rbf", C=1.0, gamma="scale")
    elif name == "svm_poly":
        model = SVC(kernel="poly", degree=3, C=1.0, gamma="scale", coef0=1.0)
    elif name == "logreg":
        model = LogisticRegression(max_iter=1000)
    elif name == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(16, 16), max_iter=2000, early_stopping=False, random_state=seed
        )
    else:
        raise ValueError(f"unknown classical model {name!r}; choose from {CLASSICAL_MODELS}")
    return make_pipeline(StandardScaler(), model)
