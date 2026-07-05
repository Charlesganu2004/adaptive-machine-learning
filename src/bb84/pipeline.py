from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from .core import NoiseType, simulate_bb84_session


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the Week 3-5 experiment."""

    transmissions_per_session: int = 128
    noisy_repetitions: int = 5
    ideal_repetitions: int = 10
    batch_size: int = 256
    base_seed: int = 42

    noise_types: tuple[NoiseType, ...] = (
        "depolarizing",
        "readout",
        "amplitude_damping",
        "combined",
    )
    noise_strengths: tuple[float, ...] = (
        0.01,
        0.03,
        0.05,
        0.08,
        0.12,
    )
    eve_options: tuple[bool, ...] = (False, True)


FEATURE_COLUMNS = [
    "eve_present",
    "sifted_key_length",
    "sifted_key_rate",
    "raw_error_count",
    "qber",
    "qber_z_basis",
    "qber_x_basis",
    "zero_to_one_error_rate",
    "one_to_zero_error_rate",
    "alice_one_rate",
    "bob_one_rate",
]


def generate_experiment_dataset(
    config: ExperimentConfig,
    output_path: Path | None = None,
    print_progress: bool = True,
) -> pd.DataFrame:
    """Run hundreds of complete sessions and optionally save the CSV."""
    rows: list[dict] = []
    run_id = 0

    for eve_present in config.eve_options:
        for repetition in range(config.ideal_repetitions):
            row = simulate_bb84_session(
                transmissions=config.transmissions_per_session,
                eve_present=eve_present,
                noise_type="ideal",
                noise_strength=0.0,
                seed=config.base_seed + run_id,
                batch_size=config.batch_size,
            )
            row["run_id"] = run_id
            row["repetition"] = repetition
            rows.append(row)
            run_id += 1

            if print_progress and run_id % 10 == 0:
                print(f"Completed {run_id} BB84 sessions.")

    for noise_type in config.noise_types:
        for strength in config.noise_strengths:
            for eve_present in config.eve_options:
                for repetition in range(config.noisy_repetitions):
                    row = simulate_bb84_session(
                        transmissions=config.transmissions_per_session,
                        eve_present=eve_present,
                        noise_type=noise_type,
                        noise_strength=float(strength),
                        seed=config.base_seed + run_id,
                        batch_size=config.batch_size,
                    )
                    row["run_id"] = run_id
                    row["repetition"] = repetition
                    rows.append(row)
                    run_id += 1

                    if print_progress and run_id % 10 == 0:
                        print(f"Completed {run_id} BB84 sessions.")

    dataframe = (
        pd.DataFrame(rows)
        .sort_values("run_id")
        .reset_index(drop=True)
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)
        if print_progress:
            print(f"Saved dataset to {output_path.resolve()}")

    return dataframe


def plot_experiment_results(
    dataframe: pd.DataFrame,
    figures_directory: Path,
) -> list[Path]:
    """Create Week 3 noise-effect graphs."""
    figures_directory.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    no_eve = dataframe[dataframe["eve_present"] == 0]
    noisy_no_eve = no_eve[no_eve["noise_type"] != "ideal"]

    qber_summary = (
        noisy_no_eve.groupby(
            ["noise_type", "noise_strength"],
            as_index=False,
        )
        .agg(
            mean_qber=("qber", "mean"),
            qber_std=("qber", "std"),
        )
        .fillna(0.0)
    )

    plt.figure(figsize=(10, 6))
    for noise_type, group in qber_summary.groupby("noise_type"):
        group = group.sort_values("noise_strength")
        plt.errorbar(
            group["noise_strength"],
            group["mean_qber"],
            yerr=group["qber_std"],
            marker="o",
            capsize=3,
            label=noise_type.replace("_", " ").title(),
        )
    plt.xlabel("Noise strength")
    plt.ylabel("Mean QBER")
    plt.title("Effect of Noise on BB84 QBER (Eve Absent)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = figures_directory / "qber_vs_noise.png"
    plt.savefig(path, dpi=300)
    plt.close()
    saved_paths.append(path)

    eve_summary = (
        dataframe.groupby(
            ["eve_present", "noise_strength"],
            as_index=False,
        )
        .agg(mean_qber=("qber", "mean"))
    )

    plt.figure(figsize=(10, 6))
    for eve_value, group in eve_summary.groupby("eve_present"):
        group = group.sort_values("noise_strength")
        label = "Eve present" if eve_value == 1 else "Eve absent"
        plt.plot(
            group["noise_strength"],
            group["mean_qber"],
            marker="o",
            label=label,
        )
    plt.xlabel("Noise strength")
    plt.ylabel("Mean QBER")
    plt.title("BB84 QBER With and Without Intercept-Resend Eve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = figures_directory / "qber_eve_comparison.png"
    plt.savefig(path, dpi=300)
    plt.close()
    saved_paths.append(path)

    damping = dataframe[
        (dataframe["noise_type"] == "amplitude_damping")
        & (dataframe["eve_present"] == 0)
    ]

    directional_summary = (
        damping.groupby("noise_strength", as_index=False)
        .agg(
            zero_to_one=("zero_to_one_error_rate", "mean"),
            one_to_zero=("one_to_zero_error_rate", "mean"),
        )
        .sort_values("noise_strength")
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        directional_summary["noise_strength"],
        directional_summary["zero_to_one"],
        marker="o",
        label="0 -> 1",
    )
    plt.plot(
        directional_summary["noise_strength"],
        directional_summary["one_to_zero"],
        marker="o",
        label="1 -> 0",
    )
    plt.xlabel("Amplitude-damping strength")
    plt.ylabel("Mean directional error rate")
    plt.title("Directional Effect of Amplitude Damping")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = figures_directory / "amplitude_damping_directionality.png"
    plt.savefig(path, dpi=300)
    plt.close()
    saved_paths.append(path)

    return saved_paths


def train_random_forest(
    dataframe: pd.DataFrame,
    models_directory: Path,
    metrics_directory: Path,
    figures_directory: Path,
    random_state: int = 42,
) -> dict:
    """Train a non-leaking Random Forest severity classifier."""
    models_directory.mkdir(parents=True, exist_ok=True)
    metrics_directory.mkdir(parents=True, exist_ok=True)
    figures_directory.mkdir(parents=True, exist_ok=True)

    X = dataframe[FEATURE_COLUMNS]
    y = dataframe["noise_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_,
    )

    metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
        "classes": model.classes_.tolist(),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "feature_columns": FEATURE_COLUMNS,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    model_path = models_directory / "bb84_noise_severity_random_forest.joblib"
    joblib.dump(model, model_path)

    metrics_path = metrics_directory / "random_forest_metrics.json"
    import json
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_path = metrics_directory / "classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_path)

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=model.classes_,
    ).plot(values_format="d")
    plt.title("Random Forest Noise-Severity Confusion Matrix")
    plt.tight_layout()
    confusion_path = figures_directory / "random_forest_confusion_matrix.png"
    plt.savefig(confusion_path, dpi=300)
    plt.close()

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS,
    ).sort_values(ascending=True)

    plt.figure(figsize=(10, 7))
    importance.plot(kind="barh")
    plt.xlabel("Random Forest feature importance")
    plt.title("BB84 Noise-Severity Feature Importance")
    plt.tight_layout()
    importance_path = figures_directory / "random_forest_feature_importance.png"
    plt.savefig(importance_path, dpi=300)
    plt.close()

    importance.sort_values(ascending=False).rename("importance").to_csv(
        metrics_directory / "feature_importance.csv",
        header=True,
    )

    metrics["model_path"] = str(model_path)
    metrics["metrics_path"] = str(metrics_path)
    metrics["confusion_figure"] = str(confusion_path)
    metrics["importance_figure"] = str(importance_path)
    return metrics
