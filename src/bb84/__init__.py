"""BB84 noisy-simulation and machine-learning package."""

from .core import (
    build_noise_model,
    build_prepare_measure_circuit,
    simulate_bb84_session,
)
from .pipeline import (
    ExperimentConfig,
    generate_experiment_dataset,
    plot_experiment_results,
    train_random_forest,
)

__all__ = [
    "build_noise_model",
    "build_prepare_measure_circuit",
    "simulate_bb84_session",
    "ExperimentConfig",
    "generate_experiment_dataset",
    "plot_experiment_results",
    "train_random_forest",
]
