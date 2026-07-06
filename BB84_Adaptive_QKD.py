# -*- coding: utf-8 -*-
"""
Adaptive Machine Learning-Assisted Noise Mitigation for BB84 QKD
================================================================

Importable module covering Weeks 2-6 of the project:
    Week 2 - Build BB84 in Qiskit
    Week 3 - Noise models + QBER-vs-noise graphs
    Week 4 - Generate the simulation dataset (CSV)
    Week 5 - Train a Random Forest noise classifier
    Week 6 - Adaptive mitigation experiments (comparison table)

Every piece of logic lives in a function, so this file can be either:
    * imported from a Jupyter notebook   ->  from BB84_Adaptive_QKD import *
    * run directly as a script           ->  python BB84_Adaptive_QKD.py

Setup:
    pip install qiskit qiskit-aer scikit-learn pandas matplotlib
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error, amplitude_damping_error, pauli_error
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score
)

# Resolve project paths from THIS file, so saving works no matter where
# Jupyter (or the shell) is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ── Shared constants ──────────────────────────────────────────
NOISE_TYPES = ["bit_flip", "phase_flip", "depolarizing", "amplitude_damping"]
STRENGTHS = [0.01, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20]

# Channel statistics used as machine-learning features.
# The two derived features carry most of the signal:
#   basis_qber_diff = qber_z - qber_x  -> +ve for bit-flip, -ve for phase-flip,
#                                          ~0 for depolarizing.
#   bit_bias        = alice_one - bob_one -> +ve for amplitude damping (|1>->|0>).
FEATURES = [
    "eve_present", "sifted_key_rate",
    "qber", "qber_z_basis", "qber_x_basis",
    "basis_qber_diff", "bit_bias",
    "alice_one_rate", "bob_one_rate",
]

# Strategy lookup: predicted noise type -> (action label, QBER reduction factor).
MITIGATION = {
    "ideal":             ("None",                               0.00),
    "bit_flip":          ("Repetition Code (3x majority vote)", 0.55),
    "phase_flip":        ("Hadamard + Repetition Code",         0.55),
    "depolarizing":      ("Increased Privacy Amplification",    0.45),
    "amplitude_damping": ("Asymmetric Threshold Decoding",      0.40),
}


# ══════════════════════════════════════════════════════════════
# WEEK 2 — Build BB84 in Qiskit
# ══════════════════════════════════════════════════════════════

def build_bb84_circuit(alice_bit, alice_basis, bob_basis):
    """Prepare one qubit in Alice's basis and measure it in Bob's basis."""
    qc = QuantumCircuit(1, 1)
    if alice_bit == 1:
        qc.x(0)          # prepare |1>
    if alice_basis == 1:
        qc.h(0)          # encode in the diagonal (X) basis
    qc.barrier()
    qc.id(0)             # quantum channel — noise is attached to this gate
    qc.barrier()
    if bob_basis == 1:
        qc.h(0)          # measure in the diagonal (X) basis
    qc.measure(0, 0)
    return qc


def _measure_one_probs(noise_model, shots=4096):
    """
    Probability that a qubit is measured as 1 for every possible
    (sent_bit, prepare_basis, measure_basis) combination, under the noise model.

    Only 8 combinations exist, so we simulate 8 circuits once instead of one
    circuit per transmission — the key speed-up for large datasets.
    """
    keys = [(bit, pb, mb) for bit in (0, 1) for pb in (0, 1) for mb in (0, 1)]
    circuits = [build_bb84_circuit(bit, pb, mb) for bit, pb, mb in keys]
    sim = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()
    result = sim.run(circuits, shots=shots).result()

    probs = {}
    for i, key in enumerate(keys):
        counts = result.get_counts(i)
        ones = sum(v for outcome, v in counts.items() if outcome.strip()[-1] == "1")
        probs[key] = ones / shots
    return probs


def run_bb84_session(n_bits, noise_model=None, eve_present=False, seed=0, shots=4096):
    """
    Simulate one complete BB84 key-exchange session.

    Returns a dict of channel statistics (the ML feature row):
    QBER, per-basis QBER, sifted-key length/rate, Alice/Bob one-rate, Eve flag.
    """
    rng = np.random.default_rng(seed)
    alice_bits  = rng.integers(0, 2, n_bits)
    alice_bases = rng.integers(0, 2, n_bits)
    bob_bases   = rng.integers(0, 2, n_bits)

    # One 8-circuit simulation gives the measurement distribution for the
    # whole session; every transmission is then a fast Bernoulli draw.
    probs = _measure_one_probs(noise_model, shots)

    if eve_present:
        # Intercept-resend attack: Eve measures in a random basis, then re-sends.
        eve_bases = rng.integers(0, 2, n_bits)
        p_eve = np.array([probs[(int(alice_bits[i]), int(alice_bases[i]), int(eve_bases[i]))]
                          for i in range(n_bits)])
        eve_bits = (rng.random(n_bits) < p_eve).astype(np.int8)
        p_bob = np.array([probs[(int(eve_bits[i]), int(eve_bases[i]), int(bob_bases[i]))]
                          for i in range(n_bits)])
    else:
        p_bob = np.array([probs[(int(alice_bits[i]), int(alice_bases[i]), int(bob_bases[i]))]
                          for i in range(n_bits)])

    bob_bits = (rng.random(n_bits) < p_bob).astype(np.int8)

    # Sift — keep only positions where Alice and Bob used the same basis.
    match        = alice_bases == bob_bases
    sifted_alice = alice_bits[match]
    sifted_bob   = bob_bits[match]
    sifted_bases = alice_bases[match]

    n_sifted = len(sifted_alice)
    errors   = sifted_alice != sifted_bob
    qber     = float(np.mean(errors)) if n_sifted else 0.0

    # Per-basis QBER — the signature that tells the noise types apart.
    z_mask = sifted_bases == 0
    x_mask = sifted_bases == 1
    qber_z = float(np.mean(errors[z_mask])) if np.any(z_mask) else 0.0
    qber_x = float(np.mean(errors[x_mask])) if np.any(x_mask) else 0.0

    alice_one = float(np.mean(sifted_alice)) if n_sifted else 0.0
    bob_one   = float(np.mean(sifted_bob))   if n_sifted else 0.0

    return {
        "eve_present":     int(eve_present),
        "sifted_length":   n_sifted,
        "sifted_key_rate": n_sifted / n_bits,
        "qber":            qber,
        "qber_z_basis":    qber_z,
        "qber_x_basis":    qber_x,
        "basis_qber_diff": qber_z - qber_x,   # bit-flip vs phase-flip
        "bit_bias":        alice_one - bob_one,  # amplitude-damping fingerprint
        "key_accuracy":    1.0 - qber,
        "alice_one_rate":  alice_one,
        "bob_one_rate":    bob_one,
    }


# ══════════════════════════════════════════════════════════════
# WEEK 3 — Noise Models and QBER Visualization
# ══════════════════════════════════════════════════════════════

def make_noise_model(noise_type, strength):
    """Build a Qiskit NoiseModel for one channel-impairment type."""
    model = NoiseModel()
    if noise_type == "bit_flip":
        err = pauli_error([("X", strength), ("I", 1 - strength)])
    elif noise_type == "phase_flip":
        err = pauli_error([("Z", strength), ("I", 1 - strength)])
    elif noise_type == "depolarizing":
        err = depolarizing_error(strength, 1)
    elif noise_type == "amplitude_damping":
        err = amplitude_damping_error(strength)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    model.add_all_qubit_quantum_error(err, ["id"])
    return model


def run_noise_analysis(strengths=None, n_bits=200, seed=42):
    """Sweep every noise type across strengths; return a tidy QBER DataFrame."""
    strengths = strengths or STRENGTHS
    rows = []
    for noise_type in NOISE_TYPES:
        for strength in strengths:
            nm = make_noise_model(noise_type, strength)
            session = run_bb84_session(n_bits, noise_model=nm, seed=seed)
            rows.append({"noise_type": noise_type, "strength": strength,
                         "qber": session["qber"]})
    return pd.DataFrame(rows)


def plot_noise_analysis(noise_df):
    """Plot QBER vs noise strength for each noise type."""
    plt.figure(figsize=(10, 5))
    for nt in NOISE_TYPES:
        sub = noise_df[noise_df["noise_type"] == nt]
        plt.plot(sub["strength"], sub["qber"], marker="o",
                 label=nt.replace("_", " ").title())
    plt.xlabel("Noise Strength")
    plt.ylabel("QBER")
    plt.title("Effect of Noise on BB84 QBER")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════
# WEEK 4 — Generate the Dataset
# ══════════════════════════════════════════════════════════════

def generate_dataset(num_samples=1000, n_bits=600, min_strength=0.04,
                     max_strength=0.25, save=True, print_progress=True):
    """Run many randomized BB84 sessions and return the feature DataFrame."""
    data = []
    for i in range(num_samples):
        rng_i = np.random.default_rng(i)
        noise_type = str(rng_i.choice(NOISE_TYPES + ["ideal"]))

        if noise_type == "ideal":
            strength, nm = 0.0, None
        else:
            strength = float(rng_i.uniform(min_strength, max_strength))
            nm = make_noise_model(noise_type, strength)

        eve = bool(rng_i.choice([False, True]))
        session = run_bb84_session(n_bits, noise_model=nm, eve_present=eve, seed=i + 1000)
        data.append({"noise_type": noise_type, "noise_strength": strength, **session})

        if print_progress and (i + 1) % 200 == 0:
            print(f"  {i + 1}/{num_samples} sessions complete")

    df = pd.DataFrame(data)
    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = DATA_DIR / "bb84_dataset.csv"
        df.to_csv(out, index=False)
        if print_progress:
            print(f"Dataset saved to {out}  (shape {df.shape})")
    return df


def plot_noise_distribution(df):
    """Bar chart of how many sessions fall under each noise type."""
    df["noise_type"].value_counts().plot(kind="bar", figsize=(8, 4))
    plt.title("Noise Type Distribution in Dataset")
    plt.xlabel("Noise Type")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════
# WEEK 5 — Train the Random Forest Classifier
# ══════════════════════════════════════════════════════════════

def train_noise_classifier(df, random_state=42):
    """Train a Random Forest to predict the noise type from channel stats."""
    X = df[FEATURES]
    y = df["noise_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training: {X_train.shape}  |  Testing: {X_test.shape}")
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return model, X_test, y_test, y_pred


def plot_confusion(df, y_test, y_pred):
    """Confusion matrix for the noise-type predictions."""
    labels = sorted(df["noise_type"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot()
    plt.title("Random Forest — Noise Type Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model):
    """Bar chart of which channel features drive the prediction."""
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    importances = importances.sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    importances.plot(kind="bar")
    plt.title("Feature Importance for Noise Classification")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return importances


# ══════════════════════════════════════════════════════════════
# WEEK 6 — Adaptive Mitigation Experiments
# ══════════════════════════════════════════════════════════════

def adaptive_experiment(model, noise_type, strength, n_bits=600, seed=99):
    """
    Adaptive mitigation pipeline for one channel condition:
      1. Run BB84 and collect channel statistics.
      2. ML model predicts the noise type from those statistics.
      3. Select the matching mitigation strategy.
      4. Re-run with reduced effective noise and measure the improvement.
    """
    nm = make_noise_model(noise_type, strength) if noise_type != "ideal" else None
    before = run_bb84_session(n_bits, noise_model=nm, seed=seed)

    features_df = pd.DataFrame([{f: before[f] for f in FEATURES}])
    predicted = model.predict(features_df)[0]

    strategy, reduction = MITIGATION.get(predicted, ("Unknown", 0.0))
    reduced_strength = strength * (1 - reduction)
    nm_after = (
        make_noise_model(noise_type, reduced_strength)
        if noise_type != "ideal" and reduced_strength > 0 else None
    )
    after = run_bb84_session(n_bits, noise_model=nm_after, seed=seed)

    return {
        "True Noise Type": noise_type,
        "Predicted Type":  predicted,
        "Strategy":        strategy,
        "QBER Before":     round(before["qber"], 4),
        "QBER After":      round(after["qber"], 4),
        "Improvement":     round(before["qber"] - after["qber"], 4),
    }


def run_mitigation_experiments(model, experiments=None):
    """Run a batch of adaptive experiments and return the comparison table."""
    experiments = experiments or [
        ("bit_flip",          0.10),
        ("phase_flip",        0.10),
        ("depolarizing",      0.10),
        ("amplitude_damping", 0.10),
        ("bit_flip",          0.20),
        ("depolarizing",      0.20),
    ]
    rows = [adaptive_experiment(model, nt, s, seed=100 + i)
            for i, (nt, s) in enumerate(experiments)]
    return pd.DataFrame(rows)


def plot_mitigation(results_df):
    """Grouped bar chart of QBER before vs after mitigation."""
    x = np.arange(len(results_df))
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, results_df["QBER Before"], width=0.4,
            label="Before Mitigation", color="steelblue")
    plt.bar(x + 0.2, results_df["QBER After"], width=0.4,
            label="After Mitigation", color="seagreen")
    plt.xticks(x, results_df["True Noise Type"], rotation=20, ha="right")
    plt.ylabel("QBER")
    plt.title("Adaptive Mitigation: QBER Before vs After")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════
# Script entry point — runs the whole pipeline end to end.
# ══════════════════════════════════════════════════════════════

def main():
    print("Week 3 — Noise analysis...")
    plot_noise_analysis(run_noise_analysis())

    print("\nWeek 4 — Generating dataset...")
    df = generate_dataset()
    plot_noise_distribution(df)

    print("\nWeek 5 — Training Random Forest...")
    model, X_test, y_test, y_pred = train_noise_classifier(df)
    plot_confusion(df, y_test, y_pred)
    plot_feature_importance(model)

    print("\nWeek 6 — Adaptive mitigation experiments...")
    results_df = run_mitigation_experiments(model)
    print(results_df.to_string(index=False))
    plot_mitigation(results_df)
    print("\nAll weeks complete.")


if __name__ == "__main__":
    main()
