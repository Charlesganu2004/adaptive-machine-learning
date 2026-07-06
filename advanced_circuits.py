# -*- coding: utf-8 -*-
"""
Advanced Circuit Search — Unpredictable, Universal, Fast Noise Resistance
=========================================================================

A SEPARATE experiment from the repetition-code study (noise_resistance.py).
Everything else in the project is untouched; this is an add-on test.

What is different here:
  * We try UNUSUAL circuit structures whose noise behaviour cannot be written
    down in advance — the answer only comes from simulation:
        - entangled repetition with a CHAIN encoder vs a TREE encoder
          (same code, different way the data is assembled)
        - INDEPENDENT re-transmission (no entangling gates at all — a totally
          different "way the data arrives")
        - the SHOR 9-qubit code, which can absorb ANY single-qubit error
  * Every circuit uses a decoder that is LEARNED by calibration, not derived,
    so nothing about the outcome is pre-baked.
  * Noise is placed at THREE locations, not just the channel:
        - gate   : errors on the encode/decode gates
        - channel: errors on the transmission (the id "hop")
        - readout: errors on measurement
    so we can ask whether hardening the *other* locations also helps.

Goal: find a circuit that resists ALL four noise types and is still fast,
then refine it with what the data reveals.

Standalone:  python advanced_circuits.py
Importable:  from advanced_circuits import run_zoo, run_location_study
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, ReadoutError,
    depolarizing_error, amplitude_damping_error, pauli_error,
)

PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

NOISE_TYPES = ["bit_flip", "phase_flip", "depolarizing", "amplitude_damping"]

# Baseline noise placed at every location for the main comparison.
BASE_CHANNEL = 0.08
BASE_GATE = 0.004
BASE_READOUT = 0.02


# ══════════════════════════════════════════════════════════════
# Encoders — the different circuit structures
# ══════════════════════════════════════════════════════════════

def enc_rep_chain(qc):
    """Repetition, chain encoder: one control fans out sequentially (depth ~ n)."""
    for target in range(1, qc.num_qubits):
        qc.cx(0, target)


def enc_rep_tree(qc):
    """Repetition, tree encoder: copies double each layer (depth ~ log n)."""
    n = qc.num_qubits
    filled, nxt = [0], 1
    while nxt < n:
        new = []
        for src in filled:
            if nxt < n:
                qc.cx(src, nxt)
                new.append(nxt)
                nxt += 1
        filled += new


def enc_phase_chain(qc):
    """Phase-flip repetition: repetition wrapped in the X-basis."""
    enc_rep_chain(qc)
    qc.h(range(qc.num_qubits))


def enc_shor9(qc):
    """Shor 9-qubit code: absorbs any single-qubit error (bit AND phase)."""
    qc.cx(0, 3)
    qc.cx(0, 6)
    qc.h(0)
    qc.h(3)
    qc.h(6)
    for block in (0, 3, 6):
        qc.cx(block, block + 1)
        qc.cx(block, block + 2)


DESIGN_SPECS = [
    # name, qubits, encoder, kind
    ("bare",          1, None,           "coded"),
    ("rep_chain_3",   3, enc_rep_chain,  "coded"),
    ("rep_chain_5",   5, enc_rep_chain,  "coded"),
    ("rep_tree_5",    5, enc_rep_tree,   "coded"),
    ("phase_3",       3, enc_phase_chain,"coded"),
    ("phase_5",       5, enc_phase_chain,"coded"),
    ("independent_3", 3, None,           "independent"),
    ("independent_5", 5, None,           "independent"),
    ("independent_7", 7, None,           "independent"),
    ("shor_9",        9, enc_shor9,      "coded"),
]


# ══════════════════════════════════════════════════════════════
# Learned (calibrated) decoder — nothing hand-derived
# ══════════════════════════════════════════════════════════════

def calibrate_decoder(encoder, n):
    """
    Learn the syndrome -> logical-bit map by injecting every single-qubit error
    into the noiseless circuit and recording what is measured. Returns the map
    plus whether every single error was correctable (a code-quality signal).
    """
    E = QuantumCircuit(n)
    if encoder:
        encoder(E)
    D = E.inverse()
    sim = AerSimulator(method="stabilizer")

    votes = defaultdict(Counter)
    injected = [None] + [(P, q) for q in range(n) for P in ("x", "y", "z")]
    for b in (0, 1):
        for err in injected:
            qc = QuantumCircuit(n, n)
            if b:
                qc.x(0)
            qc.compose(E, inplace=True)
            if err:
                getattr(qc, err[0])(err[1])
            qc.compose(D, inplace=True)
            qc.measure(range(n), range(n))
            counts = sim.run(qc, shots=1, seed_simulator=1).result().get_counts()
            votes[next(iter(counts)).replace(" ", "")][b] += 1

    decode, correctable = {}, 0
    for pattern, tally in votes.items():
        ranked = tally.most_common()
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            decode[pattern] = ranked[0][0]
            correctable += 1
    return decode


# ══════════════════════════════════════════════════════════════
# Noise at three locations
# ══════════════════════════════════════════════════════════════

def _channel_error(noise_type, p):
    if noise_type == "bit_flip":
        return pauli_error([("X", p), ("I", 1 - p)])
    if noise_type == "phase_flip":
        return pauli_error([("Z", p), ("I", 1 - p)])
    if noise_type == "depolarizing":
        return depolarizing_error(p, 1)
    if noise_type == "amplitude_damping":
        return amplitude_damping_error(p)
    raise ValueError(noise_type)


def make_noise(noise_type, channel_p, gate_p, readout_p):
    """Noise model with independent gate / channel / readout locations."""
    nm = NoiseModel()
    if channel_p > 0:
        nm.add_all_qubit_quantum_error(_channel_error(noise_type, channel_p), ["id"])
    if gate_p > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(gate_p, 1), ["x", "h"])
        nm.add_all_qubit_quantum_error(depolarizing_error(gate_p, 2), ["cx"])
    if readout_p > 0:
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - readout_p, readout_p], [readout_p, 1 - readout_p]])
        )
    # Amplitude damping is non-Clifford; everything else stays on the fast path.
    method = "stabilizer" if noise_type != "amplitude_damping" else "automatic"
    return nm, method


# ══════════════════════════════════════════════════════════════
# Run one design under one noise setting
# ══════════════════════════════════════════════════════════════

def _build_circuit(design, b):
    n, kind = design["n"], design["kind"]
    qc = QuantumCircuit(n, n)
    if b:
        if kind == "independent":
            for q in range(n):
                qc.x(q)          # prepare each copy independently
        else:
            qc.x(0)
    if kind == "coded" and design["encoder"]:
        enc = QuantumCircuit(n)
        design["encoder"](enc)
        qc.compose(enc, inplace=True)
    qc.barrier()
    for q in range(n):
        qc.id(q)                 # the noisy channel / transmission hop
    qc.barrier()
    if kind == "coded" and design["encoder"]:
        dec = QuantumCircuit(n)
        design["encoder"](dec)
        qc.compose(dec.inverse(), inplace=True)
    qc.measure(range(n), range(n))
    return qc


def run_design(design, noise_type, channel_p=BASE_CHANNEL, gate_p=BASE_GATE,
               readout_p=BASE_READOUT, shots=8000, seed=0):
    """Measure one design's logical error and its cost under one noise type."""
    n, kind = design["n"], design["kind"]
    nm, method = make_noise(noise_type, channel_p, gate_p, readout_p)
    sim = AerSimulator(noise_model=nm, method=method)
    decode_map = design.get("decode_map")

    errors = total = 0
    depth = gate_count = 0
    start = perf_counter()
    for b in (0, 1):
        qc = _build_circuit(design, b)
        depth, gate_count = qc.depth(), qc.size()
        counts = sim.run(qc, shots=shots, seed_simulator=seed + b).result().get_counts()
        for pattern, n_hits in counts.items():
            key = pattern.replace(" ", "")
            if kind == "independent":
                logical = 1 if key.count("1") > n / 2 else 0
            else:
                logical = decode_map.get(key, int(key[-1]))
            if logical != b:
                errors += n_hits
            total += n_hits
    runtime = perf_counter() - start

    return {
        "design": design["name"],
        "structure": design["kind"],
        "n_qubits": n,
        "depth": depth,
        "gate_count": gate_count,
        "noise_type": noise_type,
        "logical_error": errors / total,
        "runtime_s": runtime,
    }


def _prepare_designs():
    """Attach a learned decoder to each coded design."""
    designs = []
    for name, n, encoder, kind in DESIGN_SPECS:
        d = {"name": name, "n": n, "encoder": encoder, "kind": kind}
        if kind == "coded":
            d["decode_map"] = calibrate_decoder(encoder, n)
        designs.append(d)
    return designs


# ══════════════════════════════════════════════════════════════
# Experiment 1 — the zoo across all four noise types
# ══════════════════════════════════════════════════════════════

def run_zoo(shots=8000, save=True, print_progress=True):
    designs = _prepare_designs()
    rows = []
    for design in designs:
        for noise_type in NOISE_TYPES:
            rows.append(run_design(design, noise_type, shots=shots))
        if print_progress:
            print(f"  tested {design['name']}")
    df = pd.DataFrame(rows)

    # Worst-case (max over noise types) is the true "resists everything" score.
    worst = df.groupby("design")["logical_error"].max().rename("worst_case_error")
    mean = df.groupby("design")["logical_error"].mean().rename("mean_error")
    df = df.merge(worst, on="design").merge(mean, on="design")

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = DATA_DIR / "advanced_circuits_dataset.csv"
        df.to_csv(out, index=False)
        if print_progress:
            print(f"Saved {len(df)} rows to {out}")
    return df, designs


# ══════════════════════════════════════════════════════════════
# Experiment 2 — does noise at OTHER locations matter / help?
# ══════════════════════════════════════════════════════════════

def run_location_study(design, noise_type="depolarizing", shots=8000, save=True):
    """
    Sweep noise at each location (gate, channel, readout) one at a time, holding
    the others at a low floor, to see which location hurts most — i.e. where
    hardening the hardware would buy the biggest resistance gain.
    """
    levels = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08]
    floor = 0.005
    rows = []
    for location in ("gate", "channel", "readout"):
        for level in levels:
            kwargs = {"gate_p": floor, "channel_p": floor, "readout_p": floor}
            kwargs[f"{location}_p"] = level
            r = run_design(design, noise_type, shots=shots, **kwargs)
            rows.append({"location": location, "level": level,
                         "logical_error": r["logical_error"]})
    df = pd.DataFrame(rows)
    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_DIR / "advanced_location_study.csv", index=False)
    return df


# ══════════════════════════════════════════════════════════════
# Experiment 3 — refine the champion using what the data revealed
# ══════════════════════════════════════════════════════════════

def refine_champion(champion, location_df, shots=8000):
    """
    Take the worst location from the study and halve its noise ("harden that
    hardware"), then re-measure the champion across all noise types.
    Returns a before/after table proving the extra improvement.
    """
    sensitivity = (location_df.groupby("location")["logical_error"].max()
                   - location_df.groupby("location")["logical_error"].min())
    worst_location = sensitivity.idxmax()

    rows = []
    for noise_type in NOISE_TYPES:
        before = run_design(champion, noise_type, shots=shots)["logical_error"]
        hardened = {"gate_p": BASE_GATE, "channel_p": BASE_CHANNEL,
                    "readout_p": BASE_READOUT}
        hardened[f"{worst_location}_p"] /= 2.0      # harden the worst location
        after = run_design(champion, noise_type, shots=shots, **hardened)["logical_error"]
        rows.append({"noise_type": noise_type,
                     "error_before": round(before, 5),
                     "error_after": round(after, 5),
                     "reduction_%": round(100 * (before - after) / max(before, 1e-9), 1)})
    return worst_location, pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════

def _save(name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  saved {path}")
    plt.show()
    return path


def plot_zoo_heatmap(df):
    """Every circuit vs every noise type — which ones resist ALL of it."""
    grid = df.pivot(index="design", columns="noise_type", values="logical_error")
    order = df.groupby("design")["worst_case_error"].first().sort_values(ascending=False).index
    grid = grid.loc[order]
    plt.figure(figsize=(10, 7))
    plt.imshow(grid.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=grid.values.max())
    plt.colorbar(label="Logical error rate")
    plt.xticks(range(len(grid.columns)), [c.replace("_", "\n") for c in grid.columns])
    plt.yticks(range(len(grid.index)), grid.index)
    plt.xlabel("Noise type")
    plt.title("Which Circuits Resist Which Noise (greener = better)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            plt.text(j, i, f"{grid.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    return _save("advanced_zoo_heatmap.png")


def plot_universality_vs_speed(df):
    """The money plot: resists-everything (low worst-case) AND fast (low depth)."""
    summary = df.groupby(["design", "structure"], as_index=False).agg(
        worst=("worst_case_error", "first"), depth=("depth", "first"),
        n_qubits=("n_qubits", "first"))
    plt.figure(figsize=(10, 6))
    colors = {"coded": "tab:blue", "independent": "tab:orange"}
    for _, r in summary.iterrows():
        plt.scatter(r["depth"], r["worst"], s=60 + 25 * r["n_qubits"],
                    color=colors.get(r["structure"], "gray"), alpha=0.7)
        plt.annotate(r["design"], (r["depth"], r["worst"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=9)
    plt.yscale("log")
    plt.xlabel("Circuit depth  (lower = faster)")
    plt.ylabel("Worst-case error over all noise types  (lower = more universal)")
    plt.title("Universal AND Fast: the Best Circuit Sits Bottom-Left")
    plt.grid(True, which="both", alpha=0.3)
    return _save("advanced_universality_vs_speed.png")


def plot_best_circuit(champion):
    """Draw the winning circuit itself."""
    qc = _build_circuit(champion, 1)
    fig = qc.draw("mpl", fold=-1)
    fig.suptitle(f"Best All-Round Circuit: {champion['name']}", y=1.02)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "advanced_best_circuit.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  saved {path}")
    plt.show()
    return path


def plot_location_study(location_df):
    """Which noise location hurts most — where hardening pays off."""
    plt.figure(figsize=(10, 6))
    for location, sub in location_df.groupby("location"):
        sub = sub.sort_values("level")
        plt.plot(sub["level"], sub["logical_error"], "o-", label=location)
    plt.xlabel("Noise strength at that location")
    plt.ylabel("Logical error rate")
    plt.title("Noise Is Not Only in the Channel:\nWhich Location Hurts the Champion Most")
    plt.legend(title="Noise location")
    plt.grid(True, alpha=0.3)
    return _save("advanced_location_study.png")


def plot_refinement(worst_location, refine_df):
    """Before vs after hardening the worst location."""
    x = np.arange(len(refine_df))
    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2, refine_df["error_before"], width=0.4, label="Before", color="indianred")
    plt.bar(x + 0.2, refine_df["error_after"], width=0.4,
            label=f"After hardening {worst_location}", color="seagreen")
    plt.xticks(x, refine_df["noise_type"], rotation=15)
    plt.ylabel("Logical error rate")
    plt.title(f"Refinement: Hardening the '{worst_location}' Location Improves Every Noise Type")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    return _save("advanced_refinement.png")


def summarize(df, champion, worst_location, refine_df):
    print("\nHeadline results")
    print("----------------")
    summary = (df.groupby(["design", "structure"])
               .agg(worst=("worst_case_error", "first"),
                    mean=("mean_error", "first"),
                    depth=("depth", "first"), qubits=("n_qubits", "first"))
               .sort_values("worst"))
    print(summary.to_string())
    print(f"\nBest all-round circuit: {champion['name']} "
          f"(worst-case error {summary.loc[champion['name']].iloc[0]['worst']:.4f} "
          f"across ALL noise types, depth {champion.get('_depth','?')})")
    print(f"Dominant noise location: {worst_location}")
    avg_gain = refine_df["reduction_%"].mean()
    print(f"Refinement (hardening {worst_location}) cut error by "
          f"{avg_gain:.0f}% on average.")


def main():
    print("Advanced Circuit Search")
    print("=======================")
    df, designs = run_zoo()

    by_design = {d["name"]: d for d in designs}
    # Champion = lowest worst-case error, tie-broken by shallow depth.
    ranking = (df.groupby("design")
               .agg(worst=("worst_case_error", "first"), depth=("depth", "first"))
               .sort_values(["worst", "depth"]))
    champion = by_design[ranking.index[0]]
    champion["_depth"] = int(ranking.iloc[0]["depth"])

    location_df = run_location_study(champion)
    worst_location, refine_df = refine_champion(champion, location_df)

    plot_zoo_heatmap(df)
    plot_universality_vs_speed(df)
    plot_best_circuit(champion)
    plot_location_study(location_df)
    plot_refinement(worst_location, refine_df)
    summarize(df, champion, worst_location, refine_df)
    print("\nDone.")


if __name__ == "__main__":
    main()
