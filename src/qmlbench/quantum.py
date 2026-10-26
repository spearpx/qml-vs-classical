"""Quantum classifiers: a variational quantum classifier and a quantum-kernel SVM.

Both use the same ``ZZFeatureMap`` encoding (one qubit per feature) so their
results are comparable:

* **VQC**: feature map followed by a trainable ``RealAmplitudes`` ansatz,
  measured and read out by parity, trained with COBYLA on cross-entropy.
* **QSVC**: a classical SVM whose kernel is the state fidelity
  ``|<phi(x)|phi(x')>|^2``, estimated with compute-uncompute circuits on a
  sampler (``FidelityQuantumKernel``).

Everything runs on V2 primitives: the Qiskit reference ``StatevectorSampler``
(finite shots, no noise) by default, or an Aer ``SamplerV2`` with a noise model
(see :func:`noisy_sampler`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.primitives import BaseSamplerV2, StatevectorSampler
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.transpiler.passmanager import BasePassManager
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_machine_learning.algorithms import QSVC, VQC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.state_fidelities import ComputeUncompute

QUANTUM_MODELS = ("vqc", "qsvc")
NOISE_BASIS = ("u", "cx")


@dataclass(frozen=True)
class QuantumConfig:
    """Fixed (untuned) settings shared by every quantum run.

    Attributes:
        feature_map_reps: Repetitions of the ZZ feature map.
        ansatz_reps: Repetitions of the RealAmplitudes ansatz (VQC only).
        maxiter: COBYLA iteration budget for the VQC.
        shots: Shots per circuit for both the VQC and the kernel estimates.
    """

    feature_map_reps: int = 2
    ansatz_reps: int = 3
    maxiter: int = 100
    shots: int = 1024


def feature_map(num_features: int, reps: int = 2) -> QuantumCircuit:
    return zz_feature_map(num_features, reps=reps, entanglement="linear")


def ansatz(num_qubits: int, reps: int = 3) -> QuantumCircuit:
    return real_amplitudes(num_qubits, reps=reps, entanglement="linear")


def make_vqc(
    num_features: int,
    seed: int,
    config: QuantumConfig = QuantumConfig(),
    sampler: BaseSamplerV2 | None = None,
    pass_manager: BasePassManager | None = None,
    callback: Callable[[np.ndarray, float], None] | None = None,
) -> VQC:
    """VQC with a seeded random initial point in ``[-pi, pi]``."""
    fm = feature_map(num_features, config.feature_map_reps)
    an = ansatz(num_features, config.ansatz_reps)
    rng = np.random.default_rng(seed)
    return VQC(
        feature_map=fm,
        ansatz=an,
        optimizer=COBYLA(maxiter=config.maxiter),
        initial_point=rng.uniform(-np.pi, np.pi, an.num_parameters),
        sampler=sampler or StatevectorSampler(default_shots=config.shots, seed=seed),
        pass_manager=pass_manager,
        callback=callback,
    )


def make_qsvc(
    num_features: int,
    seed: int,
    config: QuantumConfig = QuantumConfig(),
    sampler: BaseSamplerV2 | None = None,
    pass_manager: BasePassManager | None = None,
) -> QSVC:
    """SVM (C = 1) on a sampled fidelity kernel."""
    sampler = sampler or StatevectorSampler(default_shots=config.shots, seed=seed)
    kernel = FidelityQuantumKernel(
        feature_map=feature_map(num_features, config.feature_map_reps),
        fidelity=ComputeUncompute(sampler=sampler, pass_manager=pass_manager),
    )
    return QSVC(quantum_kernel=kernel, C=1.0)


def build_noise_model(p2: float) -> NoiseModel:
    """Depolarizing noise on ``{u, cx}`` plus symmetric readout error.

    One parameter sets the scale: ``p1 = p2 / 10`` per single-qubit gate and
    readout flip probability ``p2 / 2``, roughly the ratios seen on current
    superconducting hardware.
    """
    model = NoiseModel(basis_gates=list(NOISE_BASIS))
    if p2 > 0:
        model.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["u"])
        model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
        flip = p2 / 2
        model.add_all_qubit_readout_error(ReadoutError([[1 - flip, flip], [flip, 1 - flip]]))
    return model


def noisy_sampler(p2: float, shots: int = 1024, seed: int | None = None):
    """Aer sampler with :func:`build_noise_model` and a pass manager targeting its basis.

    Returns ``(sampler, pass_manager)``; the pass manager must be given to the
    model so circuits are compiled to the gates the noise is attached to.
    """
    sampler = AerSamplerV2(
        default_shots=shots,
        seed=seed,
        options={"backend_options": {"noise_model": build_noise_model(p2)}},
    )
    pass_manager = generate_preset_pass_manager(
        optimization_level=1, basis_gates=list(NOISE_BASIS)
    )
    return sampler, pass_manager


def predict_with_sampler(
    vqc: VQC,
    X: np.ndarray,
    sampler: BaseSamplerV2,
    pass_manager: BasePassManager | None = None,
) -> np.ndarray:
    """Predict with a trained VQC's weights, executed on a different sampler.

    Used to evaluate a noiselessly trained model on noisy hardware: the same
    circuit and weights, only the execution changes. Assumes integer labels
    ``0 .. k-1`` (VQC one-hot encodes them in sorted order).
    """
    qnn = vqc.neural_network
    noisy_qnn = SamplerQNN(
        circuit=qnn.circuit,
        input_params=qnn.input_params,
        weight_params=qnn.weight_params,
        interpret=lambda x: x % vqc.num_classes,
        output_shape=vqc.num_classes,
        sampler=sampler,
        pass_manager=pass_manager,
    )
    probabilities = noisy_qnn.forward(X, vqc.weights)
    return np.argmax(probabilities, axis=1)
