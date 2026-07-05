from __future__ import annotations

from time import perf_counter
from typing import Any, Literal

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    amplitude_damping_error,
    depolarizing_error,
)

NoiseType = Literal[
    "ideal",
    "depolarizing",
    "readout",
    "amplitude_damping",
    "combined",
]


def build_prepare_measure_circuit(
    prepared_bit: int,
    preparation_basis: int,
    measurement_basis: int,
) -> QuantumCircuit:
    """Build one BB84 prepare-and-measure circuit."""
    for name, value in {
        "prepared_bit": prepared_bit,
        "preparation_basis": preparation_basis,
        "measurement_basis": measurement_basis,
    }.items():
        if value not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1.")

    circuit = QuantumCircuit(1, 1)

    if prepared_bit == 1:
        circuit.x(0)

    if preparation_basis == 1:
        circuit.h(0)

    circuit.barrier()
    circuit.id(0)
    circuit.barrier()

    if measurement_basis == 1:
        circuit.h(0)

    circuit.measure(0, 0)
    return circuit


def build_noise_model(
    noise_type: NoiseType,
    strength: float,
) -> NoiseModel | None:
    """Build a controlled Qiskit Aer noise model."""
    valid_types = {
        "ideal",
        "depolarizing",
        "readout",
        "amplitude_damping",
        "combined",
    }
    if noise_type not in valid_types:
        raise ValueError(
            f"Unsupported noise type {noise_type!r}. "
            f"Choose from {sorted(valid_types)}."
        )

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0.0 and 1.0.")

    if noise_type == "ideal" or strength == 0.0:
        return None

    model = NoiseModel()
    channel_error = None

    if noise_type in {"depolarizing", "combined"}:
        channel_error = depolarizing_error(strength, 1)

    if noise_type in {"amplitude_damping", "combined"}:
        damping_error = amplitude_damping_error(strength)
        channel_error = (
            damping_error
            if channel_error is None
            else channel_error.compose(damping_error)
        )

    if channel_error is not None:
        model.add_all_qubit_quantum_error(channel_error, ["id"])

    if noise_type in {"readout", "combined"}:
        probability = min(float(strength), 0.49)
        readout_error = ReadoutError(
            [
                [1.0 - probability, probability],
                [probability, 1.0 - probability],
            ]
        )
        model.add_all_qubit_readout_error(readout_error)

    return model


def _counts_to_bit(counts: dict[str, int]) -> int:
    measured_string = max(counts, key=counts.get)
    measured_string = measured_string.replace(" ", "")
    return int(measured_string[-1])


def _run_one_shot_circuits(
    circuits: list[QuantumCircuit],
    noise_model: NoiseModel | None,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    """Execute each circuit once and return one measured bit per circuit."""
    if not circuits:
        return np.array([], dtype=np.int8)

    simulator = (
        AerSimulator()
        if noise_model is None
        else AerSimulator(noise_model=noise_model)
    )

    measured_bits: list[int] = []

    for batch_number, start in enumerate(
        range(0, len(circuits), batch_size)
    ):
        batch = circuits[start : start + batch_size]
        batch_seed = seed + batch_number

        transpile_kwargs: dict[str, Any] = {
            "backend": simulator,
            "optimization_level": 0,
            "seed_transpiler": batch_seed,
        }
        if noise_model is not None:
            transpile_kwargs["basis_gates"] = noise_model.basis_gates

        compiled = transpile(batch, **transpile_kwargs)

        result = simulator.run(
            compiled,
            shots=1,
            seed_simulator=batch_seed,
        ).result()

        for index in range(len(compiled)):
            measured_bits.append(
                _counts_to_bit(result.get_counts(index))
            )

    return np.asarray(measured_bits, dtype=np.int8)


def _error_rate(
    expected: np.ndarray,
    observed: np.ndarray,
) -> float:
    if len(expected) == 0:
        return 0.0
    return float(np.mean(expected != observed))


def _conditional_error_rate(
    expected: np.ndarray,
    observed: np.ndarray,
    expected_value: int,
    observed_value: int,
) -> float:
    mask = expected == expected_value
    if not np.any(mask):
        return 0.0
    return float(np.mean(observed[mask] == observed_value))


def classify_noise_strength(strength: float) -> str:
    """Create the classification label used by the Random Forest."""
    if strength == 0.0:
        return "ideal"
    if strength <= 0.03:
        return "low"
    if strength <= 0.08:
        return "medium"
    return "high"


def simulate_bb84_session(
    *,
    transmissions: int,
    eve_present: bool,
    noise_type: NoiseType,
    noise_strength: float,
    seed: int,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Run one complete BB84 session and return session-level metrics."""
    if transmissions <= 0:
        raise ValueError("transmissions must be greater than zero.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    start_time = perf_counter()
    rng = np.random.default_rng(seed)

    alice_bits = rng.integers(
        0, 2, size=transmissions, dtype=np.int8
    )
    alice_bases = rng.integers(
        0, 2, size=transmissions, dtype=np.int8
    )
    bob_bases = rng.integers(
        0, 2, size=transmissions, dtype=np.int8
    )

    noise_model = build_noise_model(noise_type, noise_strength)

    if eve_present:
        eve_bases = rng.integers(
            0, 2, size=transmissions, dtype=np.int8
        )

        alice_to_eve_circuits = [
            build_prepare_measure_circuit(
                int(alice_bits[index]),
                int(alice_bases[index]),
                int(eve_bases[index]),
            )
            for index in range(transmissions)
        ]

        eve_results = _run_one_shot_circuits(
            alice_to_eve_circuits,
            noise_model,
            seed + 1_000_000,
            batch_size,
        )

        eve_to_bob_circuits = [
            build_prepare_measure_circuit(
                int(eve_results[index]),
                int(eve_bases[index]),
                int(bob_bases[index]),
            )
            for index in range(transmissions)
        ]

        bob_results = _run_one_shot_circuits(
            eve_to_bob_circuits,
            noise_model,
            seed + 2_000_000,
            batch_size,
        )
    else:
        alice_to_bob_circuits = [
            build_prepare_measure_circuit(
                int(alice_bits[index]),
                int(alice_bases[index]),
                int(bob_bases[index]),
            )
            for index in range(transmissions)
        ]

        bob_results = _run_one_shot_circuits(
            alice_to_bob_circuits,
            noise_model,
            seed + 3_000_000,
            batch_size,
        )

    matching_basis_mask = alice_bases == bob_bases
    sifted_alice = alice_bits[matching_basis_mask]
    sifted_bob = bob_results[matching_basis_mask]
    sifted_bases = alice_bases[matching_basis_mask]

    sifted_length = len(sifted_alice)
    errors = sifted_alice != sifted_bob
    qber = float(np.mean(errors)) if sifted_length else 0.0

    z_mask = sifted_bases == 0
    x_mask = sifted_bases == 1

    elapsed = perf_counter() - start_time

    return {
        "seed": int(seed),
        "eve_present": int(eve_present),
        "noise_type": noise_type,
        "noise_strength": float(noise_strength),
        "noise_class": classify_noise_strength(
            float(noise_strength)
        ),
        "transmissions": int(transmissions),
        "matching_bases": int(np.sum(matching_basis_mask)),
        "sifted_key_length": int(sifted_length),
        "sifted_key_rate": float(sifted_length / transmissions),
        "raw_error_count": int(np.sum(errors)),
        "qber": qber,
        "key_accuracy": float(1.0 - qber),
        "qber_z_basis": _error_rate(
            sifted_alice[z_mask], sifted_bob[z_mask]
        ),
        "qber_x_basis": _error_rate(
            sifted_alice[x_mask], sifted_bob[x_mask]
        ),
        "zero_to_one_error_rate": _conditional_error_rate(
            sifted_alice, sifted_bob, 0, 1
        ),
        "one_to_zero_error_rate": _conditional_error_rate(
            sifted_alice, sifted_bob, 1, 0
        ),
        "alice_one_rate": (
            float(np.mean(sifted_alice))
            if sifted_length
            else 0.0
        ),
        "bob_one_rate": (
            float(np.mean(sifted_bob))
            if sifted_length
            else 0.0
        ),
        "simulation_seconds": float(elapsed),
    }
