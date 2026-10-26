import numpy as np
import pytest

from qmlbench.datasets import DATASETS, load_split


@pytest.mark.parametrize("name, n_features, n_total", [
    ("moons", 2, 200), ("iris", 4, 100), ("breast_cancer", 4, 200),
])
def test_shapes_and_binary_labels(name, n_features, n_total):
    s = load_split(name, seed=0)
    assert s.num_features == n_features
    assert len(s.y_train) + len(s.y_test) == n_total
    assert set(np.unique(s.y_train)) == {0, 1}


@pytest.mark.parametrize("name", DATASETS)
def test_features_in_angle_range(name):
    s = load_split(name, seed=1)
    assert s.X_train.min() == pytest.approx(0) and s.X_train.max() == pytest.approx(np.pi)
    assert s.X_test.min() >= 0 and s.X_test.max() <= np.pi


@pytest.mark.parametrize("name", DATASETS)
def test_split_is_deterministic_and_seed_dependent(name):
    a, b, c = load_split(name, 3), load_split(name, 3), load_split(name, 4)
    np.testing.assert_array_equal(a.X_train, b.X_train)
    assert not np.array_equal(a.X_train, c.X_train)


def test_split_is_stratified():
    s = load_split("breast_cancer", seed=0)
    assert abs(s.y_train.mean() - s.y_test.mean()) < 0.05


def test_scaler_is_fitted_on_train_only():
    # If the test set leaked into the scaler, the train set would not span
    # exactly [0, pi] in every feature.
    s = load_split("moons", seed=2)
    np.testing.assert_allclose(s.X_train.min(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(s.X_train.max(axis=0), np.pi, atol=1e-12)


def test_unknown_dataset():
    with pytest.raises(ValueError):
        load_split("mnist", 0)
