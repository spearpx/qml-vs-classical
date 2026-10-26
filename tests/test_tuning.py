import json

import numpy as np
import pytest

from qmlbench.benchmark import evaluate, run_benchmark
from qmlbench.datasets import Split, load_split
from qmlbench.quantum import QuantumConfig
from qmlbench.tuning import TunedQSVC, TunedVQC, best_params, make_tuned

TINY = QuantumConfig(feature_map_reps=1, ansatz_reps=1, maxiter=10, shots=256)


@pytest.fixture(scope="module")
def small_split():
    s = load_split("moons", seed=0)
    return Split("moons-small", s.X_train[:30], s.X_test[:12], s.y_train[:30], s.y_test[:12])


def test_tuned_classical_selects_from_grid():
    model = make_tuned("logreg", 2, seed=0)
    evaluate(model, load_split("moons", 0))
    assert best_params(model)["C"] in [0.01, 0.1, 1, 10, 100]


def test_tuned_qsvc_cross_validates_every_grid_point(small_split):
    model = TunedQSVC(2, seed=0, config=TINY)
    result = evaluate(model, small_split)
    assert len(model.cv_results_) == 2 * 4  # 2 feature scales x 4 values of C
    assert 0 <= result["accuracy"] <= 1
    assert best_params(model)["feature_scale"] in ("[0, 1]", "[0, pi]")


def test_tuned_vqc_with_small_grid(small_split):
    model = TunedVQC(2, seed=0, config=TINY, grid={"feature_scale": [1 / np.pi], "ansatz_reps": [1]})
    evaluate(model, small_split)
    assert model.best_params_ == {"feature_scale": 1 / np.pi, "ansatz_reps": 1}
    assert model.predict(small_split.X_test).shape == (12,)


def test_benchmark_records_best_params():
    df = run_benchmark(datasets=["moons"], models=["svm_rbf"], seeds=[0])
    params = json.loads(df.best_params.iloc[0])
    assert set(params) == {"C", "gamma"}
    assert bool(df.tuned.iloc[0])
