"""Numerical checks of two scaling limits: barren plateaus and kernel concentration.

Both use exact statevector simulation (no shots), so the trends are not
artefacts of sampling noise.

* :func:`gradient_variance`: variance over random initialisations of
  ``d<O>/d theta_0`` for a ``RealAmplitudes`` ansatz whose depth grows with the
  qubit count. For a global cost (``Z`` on every qubit) the variance decays
  exponentially in the number of qubits (a barren plateau); a local cost (``Z``
  on one qubit) decays more slowly at these depths.
* :func:`kernel_concentration`: variance of the off-diagonal fidelity-kernel
  values ``|<phi(x)|phi(x')>|^2`` for random data. They concentrate
  exponentially around their (small) mean, so for many qubits the kernel matrix
  approaches the identity and an SVM trained on it cannot generalise.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from .quantum import ansatz, feature_map


def _observable(num_qubits: int, cost: str) -> SparsePauliOp:
    if cost == "global":
        return SparsePauliOp("Z" * num_qubits)
    if cost == "local":
        return SparsePauliOp("I" * (num_qubits - 1) + "Z")
    raise ValueError("cost must be 'global' or 'local'")


def gradient_variance(
    num_qubits: int,
    reps: int | None = None,
    cost: str = "global",
    samples: int = 200,
    seed: int = 0,
) -> float:
    """Variance of the parameter-shift gradient w.r.t. the first parameter.

    Args:
        num_qubits: Width of the ansatz.
        reps: Ansatz repetitions; defaults to ``num_qubits`` (depth grows with width).
        cost: ``"global"`` (``Z`` on all qubits) or ``"local"`` (``Z`` on qubit 0).
        samples: Number of random parameter vectors.
        seed: RNG seed.
    """
    circuit = ansatz(num_qubits, reps if reps is not None else num_qubits)
    obs = _observable(num_qubits, cost)
    rng = np.random.default_rng(seed)
    shift = np.zeros(circuit.num_parameters)
    shift[0] = np.pi / 2

    def energy(theta):
        return Statevector(circuit.assign_parameters(theta)).expectation_value(obs).real

    grads = []
    for _ in range(samples):
        theta = rng.uniform(0, 2 * np.pi, circuit.num_parameters)
        # RY rotations: the parameter-shift rule with shift pi/2 is exact.
        grads.append(0.5 * (energy(theta + shift) - energy(theta - shift)))
    return float(np.var(grads))


def kernel_concentration(
    num_qubits: int, num_points: int = 40, reps: int = 2, seed: int = 0
) -> tuple[float, float]:
    """Mean and variance of off-diagonal fidelity-kernel entries on random data in ``[0, pi]``."""
    fm = feature_map(num_qubits, reps)
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, np.pi, (num_points, num_qubits))
    states = np.array([Statevector(fm.assign_parameters(x)).data for x in X])
    kernel = np.abs(states.conj() @ states.T) ** 2
    off_diagonal = kernel[~np.eye(num_points, dtype=bool)]
    return float(off_diagonal.mean()), float(off_diagonal.var())
