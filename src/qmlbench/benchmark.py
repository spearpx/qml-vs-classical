"""Run every model on identical splits and collect accuracy, F1 and timings."""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .classical import CLASSICAL_MODELS, make_classical
from .datasets import DATASETS, Split, load_split
from .quantum import (
    QUANTUM_MODELS,
    QuantumConfig,
    make_qsvc,
    make_vqc,
    noisy_sampler,
    predict_with_sampler,
)
from .tuning import best_params, make_tuned

MODELS = CLASSICAL_MODELS + QUANTUM_MODELS


def make_model(name: str, num_features: int, seed: int, config: QuantumConfig = QuantumConfig()):
    if name in CLASSICAL_MODELS:
        return make_classical(name, seed)
    if name == "vqc":
        return make_vqc(num_features, seed, config)
    if name == "qsvc":
        return make_qsvc(num_features, seed, config)
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


def _scores(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def evaluate(model, split: Split) -> dict:
    """Fit on the train split, score on the test split, and time both steps.

    For quantum models the times are *classical simulation* times, not
    hardware times.
    """
    with warnings.catch_warnings():
        # qiskit-machine-learning warns that gradient-free COBYLA has no gradient.
        warnings.simplefilter("ignore", UserWarning)
        start = time.perf_counter()
        model.fit(split.X_train, split.y_train)
        train_time = time.perf_counter() - start
        start = time.perf_counter()
        y_pred = np.asarray(model.predict(split.X_test)).astype(int).ravel()
        predict_time = time.perf_counter() - start
    return {**_scores(split.y_test, y_pred), "train_time": train_time, "predict_time": predict_time}


def run_benchmark(
    datasets: Iterable[str] = DATASETS,
    models: Iterable[str] = MODELS,
    seeds: Iterable[int] = range(5),
    config: QuantumConfig = QuantumConfig(),
    tuned: bool = True,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """One row per (dataset, model, seed). Every model gets the same split per seed.

    With ``tuned=True`` (the default) every model selects its hyperparameters
    by 3-fold CV on the training split (see :mod:`qmlbench.tuning`) and
    ``train_time`` includes that search. With ``tuned=False`` the fixed
    defaults are used.
    """
    rows = []
    for dataset in datasets:
        for seed in seeds:
            split = load_split(dataset, seed)
            for name in models:
                if tuned:
                    model = make_tuned(name, split.num_features, seed, config)
                else:
                    model = make_model(name, split.num_features, seed, config)
                result = evaluate(model, split)
                params = json.dumps(best_params(model), default=str) if tuned else ""
                rows.append({"dataset": dataset, "model": name, "seed": seed, "tuned": tuned,
                             "n_train": len(split.y_train), "n_features": split.num_features,
                             **result, "best_params": params})
                if progress:
                    progress(f"{dataset:14s} seed={seed} {name:9s} acc={result['accuracy']:.3f} "
                             f"t={result['train_time']:.1f}s")
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds for each dataset and model."""
    metrics = ["accuracy", "f1", "train_time"]
    summary = results.groupby(["dataset", "model"], sort=False)[metrics].agg(["mean", "std"])
    return summary.round(4)


def noise_study(
    dataset: str = "moons",
    noise_levels: Iterable[float] = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1),
    seeds: Iterable[int] = range(3),
    config: QuantumConfig = QuantumConfig(),
    feature_scale: float = 1.0,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """VQC accuracy under depolarizing + readout noise of increasing strength.

    ``config`` and ``feature_scale`` should be the VQC setting chosen by
    tuning, so the noise study starts from the best noiseless model.

    For each seed and noise level, two scenarios are measured:

    * ``trained_ideal``: weights trained on the noiseless sampler, evaluated
      on the noisy one (a model deployed to noisy hardware);
    * ``trained_noisy``: training and evaluation both on the noisy sampler
      (the optimizer sees the noise and may partly adapt to it).
    """
    rows = []
    for seed in seeds:
        s = load_split(dataset, seed)
        split = Split(s.name, s.X_train * feature_scale, s.X_test * feature_scale, s.y_train, s.y_test)
        ideal = make_vqc(split.num_features, seed, config)
        base = evaluate(ideal, split)
        for p2 in noise_levels:
            sampler, pm = noisy_sampler(p2, config.shots, seed)
            y_pred = predict_with_sampler(ideal, split.X_test, sampler, pm)
            rows.append({"seed": seed, "p2": p2, "scenario": "trained_ideal",
                         **_scores(split.y_test, y_pred), "ideal_accuracy": base["accuracy"]})

            noisy = make_vqc(split.num_features, seed, config, sampler=sampler, pass_manager=pm)
            result = evaluate(noisy, split)
            rows.append({"seed": seed, "p2": p2, "scenario": "trained_noisy",
                         "accuracy": result["accuracy"], "f1": result["f1"],
                         "ideal_accuracy": base["accuracy"]})
            if progress:
                progress(f"noise seed={seed} p2={p2:.3f} ideal-trained={rows[-2]['accuracy']:.3f} "
                         f"noisy-trained={result['accuracy']:.3f}")
    return pd.DataFrame(rows)
