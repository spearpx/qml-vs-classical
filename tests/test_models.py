import numpy as np
import pytest

from qmlbench.benchmark import evaluate, make_model, noise_study, run_benchmark, summarize
from qmlbench.classical import CLASSICAL_MODELS
from qmlbench.datasets import Split, load_split
from qmlbench.quantum import QuantumConfig, make_vqc, noisy_sampler, predict_with_sampler

TINY = QuantumConfig(feature_map_reps=1, ansatz_reps=1, maxiter=15, shots=256)


@pytest.fixture(scope="module")
def small_split():
    s = load_split("moons", seed=0)
    return Split("moons-small", s.X_train[:24], s.X_test[:12], s.y_train[:24], s.y_test[:12])


@pytest.mark.parametrize("name", CLASSICAL_MODELS)
def test_classical_models_learn_moons(name):
    result = evaluate(make_model(name, 2, seed=0), load_split("moons", 0))
    assert result["accuracy"] > 0.75
    assert set(result) == {"accuracy", "f1", "train_time", "predict_time"}


@pytest.mark.parametrize("name", ["vqc", "qsvc"])
def test_quantum_models_run(name, small_split):
    result = evaluate(make_model(name, 2, seed=0, config=TINY), small_split)
    assert 0 <= result["accuracy"] <= 1
    assert result["train_time"] > 0


def test_vqc_is_reproducible(small_split):
    a = evaluate(make_vqc(2, seed=5, config=TINY), small_split)
    b = evaluate(make_vqc(2, seed=5, config=TINY), small_split)
    assert a["accuracy"] == b["accuracy"]


def test_predict_with_noiseless_sampler_matches_vqc(small_split):
    vqc = make_vqc(2, seed=1, config=TINY)
    vqc.fit(small_split.X_train, small_split.y_train)
    sampler, pm = noisy_sampler(0.0, shots=4096, seed=1)
    ours = predict_with_sampler(vqc, small_split.X_test, sampler, pm)
    theirs = np.asarray(vqc.predict(small_split.X_test)).astype(int).ravel()
    # Both are shot-based; agreement should be near-perfect, not necessarily exact.
    assert np.mean(ours == theirs) >= 0.9


def test_run_benchmark_and_summary():
    df = run_benchmark(datasets=["moons"], models=["logreg", "svm_rbf"], seeds=[0, 1])
    assert len(df) == 4
    summary = summarize(df)
    assert ("accuracy", "mean") in summary.columns


def test_noise_study_shape(monkeypatch):
    import qmlbench.benchmark as bench

    def tiny_split(name, seed):
        s = load_split(name, seed)
        return Split("tiny", s.X_train[:16], s.X_test[:8], s.y_train[:16], s.y_test[:8])

    monkeypatch.setattr(bench, "load_split", tiny_split)
    df = noise_study(noise_levels=(0.0, 0.05), seeds=[0], config=TINY)
    assert len(df) == 4
    assert set(df.scenario) == {"trained_ideal", "trained_noisy"}


def test_unknown_model():
    with pytest.raises(ValueError):
        make_model("random_forest", 2, 0)
