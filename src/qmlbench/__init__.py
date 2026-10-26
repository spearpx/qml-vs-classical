"""Honest benchmark of quantum classifiers against classical baselines."""

from .benchmark import MODELS, evaluate, make_model, noise_study, run_benchmark, summarize
from .classical import CLASSICAL_MODELS, make_classical
from .datasets import DATASETS, Split, load_split
from .quantum import QUANTUM_MODELS, QuantumConfig, make_qsvc, make_vqc, noisy_sampler
from .scaling import gradient_variance, kernel_concentration

__all__ = [
    "CLASSICAL_MODELS",
    "DATASETS",
    "MODELS",
    "QUANTUM_MODELS",
    "QuantumConfig",
    "Split",
    "evaluate",
    "gradient_variance",
    "kernel_concentration",
    "load_split",
    "make_classical",
    "make_model",
    "make_qsvc",
    "make_vqc",
    "noise_study",
    "noisy_sampler",
    "run_benchmark",
    "summarize",
]
