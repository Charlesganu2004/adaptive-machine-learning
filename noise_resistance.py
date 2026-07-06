# -*- coding: utf-8 -*-
"""
Circuit Hardening & Noise-Resistance Experiments for BB84 QKD
=============================================================

Goal: instead of only *classifying* noise, actively *strengthen* the quantum
circuit against it with error-correcting codes, and measure two things:

    1. How much the logical error rate drops  (the noise resistance / breakthrough)
    2. What it costs in circuit depth and runtime (does hardening make it slower?)

We try a whole "circuit zoo" (60+ variants):
    * bit-flip repetition code   at distances 1,3,5,...,15   (protects X errors)
    * phase-flip repetition code at distances 1,3,5,...,11   (protects Z errors)
    * every code against the noise it is designed for  (matched -> huge gains)
    * codes against the WRONG noise                    (mismatched -> no help)

Every experiment is a real Qiskit circuit simulated under a noise model; the
results are saved as a CSV dataset and a set of figures.

Importable:  from noise_resistance import run_resistance_experiments, make_all_figures
Standalone:  python noise_resistance.py
"""

from __future__ import annotations

import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from BB84_Adaptive_QKD import make_noise_model   # reuse the same noise models

PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Distances to try (odd only, so majority vote never ties).
BITFLIP_DISTANCES = [1, 3, 5, 7, 9, 11, 13, 15]
PHASEFLIP_DISTANCES = [1, 3, 5, 7, 9, 11]
STRENGTHS = [0.05, 0.10, 0.15, 0.20]


# ══════════════════════════════════════════════════════════════
# Circuit builders — the "hardened" circuits
# ══════════════════════════════════════════════════════════════

def build_repetition_circuit(bit, distance, code="bitflip"):
    """
    Encode one logical bit into `distance` physical qubits, send them through a
    noisy channel (the id gates carry the noise), then measure for a majority
    vote.  code="phaseflip" wraps the same idea in the X-basis (H...H) so it
    protects against phase (Z) errors instead of bit (X) errors.
    """
    qc = QuantumCircuit(distance, distance)

    if bit == 1:
        qc.x(0)
    for target in range(1, distance):
        qc.cx(0, target)          # copy the bit onto every physical qubit

    if code == "phaseflip":
        qc.h(range(distance))     # move into the X-basis

    qc.barrier()
    for q in range(distance):
        qc.id(q)                  # noisy quantum channel (one hop per qubit)
    qc.barrier()

    if code == "phaseflip":
        qc.h(range(distance))     # back to the computational basis

    qc.measure(range(distance), range(distance))
    return qc


def _majority_vote(bitstring):
    """Decode a measured physical string to a single logical bit."""
    ones = bitstring.count("1")
    return 1 if ones > len(bitstring) / 2 else 0


def analytic_logical_error(distance, p):
    """
    Exact logical error rate of a distance-d repetition code under independent
    flip probability p (majority vote). Lets us draw smooth curves below the
    finite-shot measurement floor.
    """
    threshold = distance // 2 + 1
    return float(sum(comb(distance, k) * p**k * (1 - p) ** (distance - k)
                     for k in range(threshold, distance + 1)))


# ══════════════════════════════════════════════════════════════
# One experiment
# ══════════════════════════════════════════════════════════════

def measure_logical_error(distance, code, noise_type, strength,
                          shots=20000, seed=0):
    """Run one hardened circuit and return its logical error rate + cost."""
    noise_model = make_noise_model(noise_type, strength)
    # Bit/phase-flip and depolarizing noise is Clifford, so the fast stabilizer
    # simulator handles even 15-qubit codes. Amplitude damping is non-Clifford,
    # so fall back to the automatic (density-matrix) method for it.
    clifford = noise_type in ("bit_flip", "phase_flip", "depolarizing")
    simulator = AerSimulator(
        noise_model=noise_model,
        method="stabilizer" if clifford else "automatic",
    )

    errors = total = 0
    depth = gate_count = n_qubits = 0

    start = time.perf_counter()
    for bit in (0, 1):                          # send both logical values
        circuit = build_repetition_circuit(bit, distance, code)
        depth, gate_count, n_qubits = circuit.depth(), circuit.size(), circuit.num_qubits
        result = simulator.run(circuit, shots=shots, seed_simulator=seed + bit).result()
        for outcome, n in result.get_counts().items():
            if _majority_vote(outcome.replace(" ", "")) != bit:
                errors += n
            total += n
    runtime = time.perf_counter() - start

    matched = (code == "bitflip" and noise_type == "bit_flip") or \
              (code == "phaseflip" and noise_type == "phase_flip")

    return {
        "code": code,
        "distance": distance,
        "n_qubits": n_qubits,
        "depth": depth,
        "gate_count": gate_count,
        "noise_type": noise_type,
        "strength": strength,
        "matched": matched,
        "logical_qber": errors / total,
        "analytic_qber": analytic_logical_error(distance, strength) if matched else np.nan,
        "runtime_s": runtime,
        "shots": shots,
    }


# ══════════════════════════════════════════════════════════════
# The full experiment grid (60+ circuits)
# ══════════════════════════════════════════════════════════════

def run_resistance_experiments(shots=20000, save=True, print_progress=True):
    """Try every circuit in the zoo and return the results DataFrame."""
    rows = []

    def go(distance, code, noise_type, strength):
        rows.append(measure_logical_error(distance, code, noise_type, strength, shots))
        if print_progress and len(rows) % 15 == 0:
            print(f"  {len(rows)} circuits tested")

    # Matched: bit-flip code vs bit-flip noise
    for d in BITFLIP_DISTANCES:
        for p in STRENGTHS:
            go(d, "bitflip", "bit_flip", p)

    # Matched: phase-flip code vs phase-flip noise
    for d in PHASEFLIP_DISTANCES:
        for p in STRENGTHS:
            go(d, "phaseflip", "phase_flip", p)

    # Mismatched: right code, wrong noise (should NOT help)
    for d in [3, 7, 11]:
        go(d, "bitflip", "phase_flip", 0.10)
        go(d, "phaseflip", "bit_flip", 0.10)

    df = pd.DataFrame(rows)

    # Improvement factor = how many times lower the error is vs the bare qubit
    # (distance-1) of the same code at the same strength. Only meaningful for
    # matched code/noise pairs, so leave the rest NaN.
    baseline = (
        df[df["distance"] == 1]
        .set_index(["code", "strength"])["logical_qber"]
        .to_dict()
    )
    floor = 1.0 / (2 * shots)                    # finite-shot resolution
    df["physical_qber"] = df.apply(
        lambda r: baseline.get((r["code"], r["strength"]), np.nan), axis=1
    )
    df["improvement_factor"] = np.where(
        df["matched"],
        df["physical_qber"] / df["logical_qber"].clip(lower=floor),
        np.nan,
    )
    # Exact (analytic) improvement — not limited by the shot floor, so it shows
    # the true order-of-magnitude suppression.
    df["analytic_improvement"] = np.where(
        df["matched"], df["strength"] / df["analytic_qber"].clip(lower=1e-15), np.nan
    )

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = DATA_DIR / "noise_resistance_dataset.csv"
        df.to_csv(out, index=False)
        if print_progress:
            print(f"Saved {len(df)} circuit experiments to {out}")

    return df


# ══════════════════════════════════════════════════════════════
# Figures — the pictures that tell the story
# ══════════════════════════════════════════════════════════════

def _save(fig_name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / fig_name
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  saved {path}")
    plt.show()
    return path


def plot_logical_vs_distance(df):
    """Breakthrough plot: logical error crashes as the code gets stronger."""
    data = df[(df["code"] == "bitflip") & (df["matched"])]
    distances = sorted(data["distance"].unique())
    plt.figure(figsize=(10, 6))
    for p in sorted(data["strength"].unique()):
        smooth = [analytic_logical_error(d, p) for d in distances]
        plt.plot(distances, smooth, "--", alpha=0.5)
        sub = data[data["strength"] == p].sort_values("distance")
        plt.plot(sub["distance"], sub["logical_qber"].clip(lower=1e-6),
                 "o-", label=f"physical p = {p}")
    plt.yscale("log")
    plt.xlabel("Code distance (physical qubits per logical bit)")
    plt.ylabel("Logical error rate (log scale)")
    plt.title("Strengthening the Circuit Crushes the Error Rate\n"
              "Bit-flip repetition code vs bit-flip noise (dashed = theory)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    return _save("hardening_logical_vs_distance.png")


def plot_improvement_factor(df):
    """How many times better the hardened circuit is than a bare qubit."""
    data = df[(df["code"] == "bitflip") & (df["matched"])]
    plt.figure(figsize=(10, 6))
    for p in sorted(data["strength"].unique()):
        sub = data[data["strength"] == p].sort_values("distance")
        line, = plt.plot(sub["distance"], sub["analytic_improvement"],
                         "-", label=f"p = {p} (exact)")
        # Simulation points confirm the theory down to the shot-noise floor.
        plt.plot(sub["distance"], sub["improvement_factor"], "o",
                 color=line.get_color(), alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Code distance")
    plt.ylabel("Improvement factor  (physical / logical error)")
    plt.title("Noise-Resistance Improvement vs Circuit Strength\n"
              "lines = exact theory, dots = simulation")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    return _save("hardening_improvement_factor.png")


def plot_speed_cost(df):
    """The trade-off: a stronger circuit is deeper and uses more qubits, so it
    takes longer to execute on real hardware (execution time ~ depth)."""
    data = (df[(df["code"] == "bitflip") & (df["matched"])]
            .groupby("distance", as_index=False)
            .agg(depth=("depth", "mean"), gate_count=("gate_count", "mean"),
                 n_qubits=("n_qubits", "mean")))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(data["distance"], data["depth"], "o-", color="tab:blue",
             label="Circuit depth")
    ax1.set_xlabel("Code distance")
    ax1.set_ylabel("Circuit depth = gate layers (~ execution time)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(data["distance"], data["n_qubits"], "s--", color="tab:red",
             label="Physical qubits")
    ax2.set_ylabel("Physical qubits used", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    plt.title("Cost of Strengthening: Deeper Circuits, More Qubits\n"
              "(both grow linearly -> execution scales with code distance)")
    ax1.grid(True, alpha=0.3)
    return _save("hardening_speed_cost.png")


def plot_tradeoff(df):
    """Is it worth it? Error suppression vs the circuit depth you pay for it."""
    data = df[(df["code"] == "bitflip") & (df["matched"]) & (df["strength"] == 0.10)]
    data = data.sort_values("distance")
    plt.figure(figsize=(10, 6))
    plt.plot(data["depth"], data["analytic_improvement"], "o-")
    for _, r in data.iterrows():
        plt.annotate(f"d={int(r['distance'])}", (r["depth"], r["analytic_improvement"]),
                     textcoords="offset points", xytext=(6, 4))
    plt.yscale("log")
    plt.xlabel("Circuit depth (proxy for hardware execution time)")
    plt.ylabel("Improvement factor (log)")
    plt.title("Return on Investment of Circuit Hardening (noise p = 0.10)\n"
              "error falls exponentially while cost grows only linearly")
    plt.grid(True, which="both", alpha=0.3)
    return _save("hardening_tradeoff.png")


def plot_both_codes(df):
    """Both codes we tried deliver huge gains across every noise level."""
    d = 9
    plt.figure(figsize=(10, 6))
    for code, label, marker in [("bitflip", "Bit-flip code vs bit noise", "o-"),
                                ("phaseflip", "Phase-flip code vs phase noise", "s-")]:
        sub = (df[(df["code"] == code) & (df["matched"]) & (df["distance"] == d)]
               .sort_values("strength"))
        plt.plot(sub["strength"], sub["analytic_improvement"], marker, label=label)
    plt.yscale("log")
    plt.xlabel("Physical noise strength")
    plt.ylabel("Improvement factor (log)")
    plt.title(f"Both Hardened Circuits Beat the Noise (code distance {d})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    return _save("hardening_both_codes.png")


def plot_heatmap(df):
    """Logical error rate across code distance and noise strength."""
    data = df[(df["code"] == "bitflip") & (df["matched"])]
    grid = data.pivot(index="strength", columns="distance", values="logical_qber")
    plt.figure(figsize=(10, 6))
    plt.imshow(grid.values, aspect="auto", origin="lower", cmap="viridis_r")
    plt.colorbar(label="Logical error rate")
    plt.xticks(range(len(grid.columns)), grid.columns)
    plt.yticks(range(len(grid.index)), grid.index)
    plt.xlabel("Code distance")
    plt.ylabel("Noise strength")
    plt.title("Logical Error Rate: Stronger Codes (right) Beat Stronger Noise (up)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            plt.text(j, i, f"{v:.3f}", ha="center", va="center",
                     color="white" if v > grid.values.mean() else "black", fontsize=8)
    return _save("hardening_heatmap.png")


def make_all_figures(df):
    """Produce every figure and return the saved paths."""
    return [
        plot_logical_vs_distance(df),
        plot_improvement_factor(df),
        plot_speed_cost(df),
        plot_tradeoff(df),
        plot_both_codes(df),
        plot_heatmap(df),
    ]


def summarize(df):
    """Print the headline breakthrough numbers."""
    matched = df[df["matched"]]
    best = matched.loc[matched["analytic_improvement"].idxmax()]
    print("\nHeadline results")
    print("----------------")
    print(f"Circuits tested: {len(df)}")
    print(f"Best exact improvement: {best['analytic_improvement']:,.0f}x  "
          f"({best['code']} code, distance {int(best['distance'])}, "
          f"noise p={best['strength']})")
    print(f"  physical error {best['strength']:.3f} -> "
          f"logical error {best['analytic_qber']:.2e}")
    sim_best = matched.loc[matched["improvement_factor"].idxmax()]
    print(f"Best measured improvement (shot-limited): "
          f"{sim_best['improvement_factor']:.0f}x")
    deepest = df.loc[df["depth"].idxmax()]
    shallow = df[df["distance"] == 1].iloc[0]
    print(f"Speed/size cost: depth {int(shallow['depth'])} -> {int(deepest['depth'])}, "
          f"qubits {int(shallow['n_qubits'])} -> {int(deepest['n_qubits'])} "
          f"(d=1 -> d={int(deepest['distance'])}) — grows only linearly.")


# ══════════════════════════════════════════════════════════════
# Closing the loop — ML prediction SELECTS the mitigation (the abstract)
# ══════════════════════════════════════════════════════════════

# Which error-correcting code defends against each predicted noise condition.
# Phase-flip noise corrupts X-basis information -> phase-flip code.
# Everything else with bit-visible errors (bit-flip, depolarizing, amplitude
# damping) -> bit-flip repetition code.  Ideal -> no action needed.
CODE_FOR_NOISE = {
    "bit_flip": "bitflip",
    "phase_flip": "phaseflip",
    "depolarizing": "bitflip",
    "amplitude_damping": "bitflip",
    "ideal": None,
}


def adaptive_mitigation(model, environments=None, code_distance=9, shots=20000):
    """
    The full adaptive framework from the abstract, end to end:

        measure the noisy channel  ->  ML predicts the noise condition
        ->  framework SELECTS the matching error-correcting code
        ->  re-run protected and compare QBER.

    Returns a comparison table across noise environments.
    """
    from BB84_Adaptive_QKD import run_bb84_session, FEATURES

    environments = environments or [
        ("bit_flip", 0.10), ("phase_flip", 0.10), ("depolarizing", 0.10),
        ("amplitude_damping", 0.10), ("bit_flip", 0.20), ("phase_flip", 0.20),
    ]
    floor = 1.0 / (2 * shots)
    rows = []
    for noise_type, strength in environments:
        noise_model = make_noise_model(noise_type, strength)

        # 1. Measure the channel with an unprotected BB84 session.
        session = run_bb84_session(600, noise_model=noise_model, seed=7)
        qber_before = session["qber"]

        # 2. ML predicts the noise condition from the channel statistics.
        features = pd.DataFrame([{f: session[f] for f in FEATURES}])
        predicted = model.predict(features)[0]

        # 3. The framework selects the mitigation action for that prediction.
        code = CODE_FOR_NOISE.get(predicted, "bitflip")

        # 4. Apply it and measure the protected QBER.
        if code is None:
            qber_after, action = qber_before, "None (channel looks clean)"
        else:
            result = measure_logical_error(code_distance, code, noise_type, strength, shots)
            qber_after = result["logical_qber"]
            action = f"{code} code, distance {code_distance}"

        rows.append({
            "noise_environment": f"{noise_type} (p={strength})",
            "true_noise": noise_type,
            "ml_predicted": predicted,
            "selected_mitigation": action,
            "qber_before": round(qber_before, 4),
            "qber_after": round(qber_after, 5),
            "improvement_x": round(qber_before / max(qber_after, floor), 1),
        })
    return pd.DataFrame(rows)


def plot_adaptive_mitigation(adaptive_df):
    """QBER before vs after the ML-selected mitigation, per noise environment."""
    x = np.arange(len(adaptive_df))
    before = adaptive_df["qber_before"].clip(lower=1e-5)
    after = adaptive_df["qber_after"].clip(lower=1e-5)
    plt.figure(figsize=(11, 6))
    plt.bar(x - 0.2, before, width=0.4, label="QBER before (unprotected)", color="indianred")
    plt.bar(x + 0.2, after, width=0.4, label="QBER after (ML-selected code)", color="seagreen")
    plt.yscale("log")
    plt.xticks(x, adaptive_df["noise_environment"], rotation=20, ha="right")
    plt.ylabel("QBER (log scale)")
    plt.title("Adaptive Framework: ML Predicts the Noise, Then Picks the Fix")
    plt.legend()
    plt.grid(True, axis="y", which="both", alpha=0.3)
    return _save("adaptive_ml_selected_mitigation.png")


def main():
    print("Circuit Hardening Experiments")
    print("=============================")
    df = run_resistance_experiments()
    make_all_figures(df)
    summarize(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
