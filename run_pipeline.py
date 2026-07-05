from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from bb84.pipeline import (  # noqa: E402
    ExperimentConfig,
    generate_experiment_dataset,
    plot_experiment_results,
    train_random_forest,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run BB84 noise experiments, generate a CSV, "
            "create graphs, and train a Random Forest."
        )
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run a small validation experiment before the full "
            "hundreds-of-simulations experiment."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.quick:
        config = ExperimentConfig(
            transmissions_per_session=32,
            noisy_repetitions=1,
            ideal_repetitions=2,
            noise_strengths=(0.03, 0.08, 0.12),
        )
        dataset_name = "bb84_experiment_dataset_quick.csv"
    else:
        config = ExperimentConfig()
        dataset_name = "bb84_experiment_dataset.csv"

    dataset_path = PROJECT_ROOT / "data" / "processed" / dataset_name
    figures_directory = PROJECT_ROOT / "results" / "figures"
    models_directory = PROJECT_ROOT / "results" / "models"
    metrics_directory = PROJECT_ROOT / "results" / "metrics"

    print("BB84 Weeks 2-5 Pipeline")
    print("=======================")
    print(f"Quick mode: {args.quick}")
    print(
        "Transmissions per session: "
        f"{config.transmissions_per_session}"
    )
    print()

    dataframe = generate_experiment_dataset(
        config,
        output_path=dataset_path,
        print_progress=True,
    )

    figure_paths = plot_experiment_results(
        dataframe,
        figures_directory,
    )

    metrics = train_random_forest(
        dataframe,
        models_directory,
        metrics_directory,
        figures_directory,
        random_state=config.base_seed,
    )

    print()
    print("Pipeline complete")
    print("-----------------")
    print(f"Dataset rows: {len(dataframe)}")
    print(f"Dataset: {dataset_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(
        "Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("Figures:")
    for path in figure_paths:
        print(f"  - {path}")
    print(f"  - {metrics['confusion_figure']}")
    print(f"  - {metrics['importance_figure']}")


if __name__ == "__main__":
    main()
