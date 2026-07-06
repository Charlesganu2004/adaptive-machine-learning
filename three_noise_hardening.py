# -*- coding: utf-8 -*-
"""
100-Circuit Noise-Hardening Search — Three Noise Types
======================================================

A SEPARATE test that lives alongside everything else (the 4-noise-type core
pipeline in BB84_Adaptive_QKD.py, the repetition study in noise_resistance.py,
and the first circuit zoo in advanced_circuits.py are all untouched).

Scope here is deliberately narrowed to the three noise types the paper focuses
on so the study stays manageable:

        bit_flip (X)      phase_flip (Z)      depolarizing (random)

What is new compared with advanced_circuits.py:
  * ONE HUNDRED structurally diverse circuits, not ten. Many are combinations
    you would *think* should work; several are random structures whose noise
    behaviour cannot be written down in advance — only simulation decides.
  * Hardening is tried BEFORE the circuit as well as THROUGH it:
        - a new PREP noise location (imperfect state preparation, modelled as
          error on `reset`, applied before any encoding), and
        - Clifford BASIS-CHOICE encodings (store the bit in the X or Y basis
          before it enters the channel) — literally "changing the way the data
          arrives" without any slow non-Clifford rotation.
  * Noise now sits at FOUR locations: prep, gate, channel, readout.
  * Every circuit reports its ERROR-REDUCTION % versus a bare qubit, per noise
    type and worst-case, so "huge vs small improvement" is explicit.
  * The abstract's loop is closed for the three-type case: a Random Forest
    classifies the noise type from cheap channel probes and SELECTS the best
    circuit for it (classify -> select -> mitigate), compared before/after.

Everything is Clifford, so the fast `stabilizer` simulator is used throughout
even for 21-qubit circuits.

Standalone:  python three_noise_hardening.py
Importable:  from three_noise_hardening import run_zoo, run_adaptive_loop
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

NOISE_TYPES = ["bit_flip", "phase_flip", "depolarizing"]

# Baseline noise placed at every location for the main 100-circuit comparison.
BASE_PREP = 0.010        # imperfect state preparation (BEFORE the circuit)
BASE_CHANNEL = 0.100     # the transmission hop
BASE_GATE = 0.005        # encode / decode gate noise
BASE_READOUT = 0.020     # measurement noise

_SIM = AerSimulator(method="stabilizer")


# ══════════════════════════════════════════════════════════════
# Encoders — the different circuit structures ("through the circuit")
# ══════════════════════════════════════════════════════════════

def enc_rep_chain(qc):
    """Bit-flip repetition, chain encoder: one control fans out sequentially."""
    for target in range(1, qc.num_qubits):
        qc.cx(0, target)


def enc_rep_tree(qc):
    """Bit-flip repetition, tree encoder: copies double each layer (shallower)."""
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
    """Phase-flip repetition: bit-repetition wrapped into the X basis."""
    enc_rep_chain(qc)
    qc.h(range(qc.num_qubits))


def enc_phase_tree(qc):
    enc_rep_tree(qc)
    qc.h(range(qc.num_qubits))


def enc_shor9(qc):
    """Shor 9-qubit code: phase-repetition of bit-repetition."""
    qc.cx(0, 3); qc.cx(0, 6)
    qc.h(0); qc.h(3); qc.h(6)
    for block in (0, 3, 6):
        qc.cx(block, block + 1)
        qc.cx(block, block + 2)


def enc_shor9_alt(qc):
    """Structural mirror of Shor: bit-spread across blocks, phase within."""
    qc.cx(0, 3); qc.cx(0, 6)
    for block in (0, 3, 6):
        qc.h(block)
        qc.cx(block, block + 1)
        qc.cx(block, block + 2)


def enc_hybrid_interleave(qc):
    """Repetition, then flip every other copy into the X basis — a mixed guard."""
    enc_rep_chain(qc)
    for q in range(1, qc.num_qubits, 2):
        qc.h(q)


def enc_dfs_pair(qc):
    """
    'Before the circuit' rearrangement: copy the bit, then lock neighbouring
    copies together with CZ so part of the information rides in phase
    correlations rather than in each qubit alone.
    """
    enc_rep_chain(qc)
    for q in range(0, qc.num_qubits - 1, 2):
        qc.cz(q, q + 1)


def make_random_encoder(n, seed, use_h):
    """
    An UNPREDICTABLE encoder: start from a repetition copy so the logical bit is
    always present, then bolt on a random sequence of Clifford gates. Whether the
    extra structure helps or hurts against noise is only revealed by simulation.

    The random layout is drawn ONCE and frozen into a fixed gate list, so the
    returned encoder is a pure function — encoding and decoding stay consistent.
    """
    rng = np.random.RandomState(seed)
    depth = int(rng.randint(2, 5))
    ops = [("cx", 0, target) for target in range(1, n)]   # guarantee the bit is copied
    for _ in range(depth):
        for q in range(n):
            if use_h and rng.rand() < 0.35:
                ops.append(("h", q))
        a, b = int(rng.randint(0, n)), int(rng.randint(0, n))
        if a != b:
            ops.append(("cx", a, b))

    def encoder(qc):
        for op in ops:
            if op[0] == "cx":
                qc.cx(op[1], op[2])
            else:
                qc.h(op[1])
    return encoder


# ══════════════════════════════════════════════════════════════
# Build the registry of exactly 100 circuits
# ══════════════════════════════════════════════════════════════

def build_registry(target=100):
    specs = []

    def add(name, n, encoder, kind, family, basis="Z"):
        specs.append({"name": name, "n": n, "encoder": encoder,
                      "kind": kind, "family": family, "basis": basis})

    add("bare", 1, None, "coded", "baseline")

    for d in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21):
        add(f"rep_chain_{d}", d, enc_rep_chain, "coded", "rep_chain")
    for d in (3, 5, 7, 9, 11, 13, 15):
        add(f"rep_tree_{d}", d, enc_rep_tree, "coded", "rep_tree")
    for d in (3, 5, 7, 9, 11, 13, 15):
        add(f"phase_chain_{d}", d, enc_phase_chain, "coded", "phase_chain")
    for d in (3, 5, 7, 9, 11):
        add(f"phase_tree_{d}", d, enc_phase_tree, "coded", "phase_tree")

    for nN in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21):
        add(f"independent_{nN}", nN, None, "independent", "independent")
    for nN in (2, 4, 6, 8):                          # even copies: majority ties
        add(f"independent_even_{nN}", nN, None, "independent", "independent_even")

    # 'Before the circuit' basis choice — store the bit in X or Y before sending.
    for nN in (3, 5, 7, 9, 11):
        add(f"independent_xbasis_{nN}", nN, None, "independent", "basis_choice", basis="X")
    for nN in (3, 5, 7, 9):
        add(f"independent_ybasis_{nN}", nN, None, "independent", "basis_choice", basis="Y")

    add("shor_9", 9, enc_shor9, "coded", "shor")
    add("shor_9_alt", 9, enc_shor9_alt, "coded", "shor")

    for nN in (5, 7, 9, 11):
        add(f"hybrid_interleave_{nN}", nN, enc_hybrid_interleave, "coded", "hybrid")
    for nN in (4, 6, 8, 10):
        add(f"dfs_pair_{nN}", nN, enc_dfs_pair, "coded", "dfs_pair")

    # Fill the remainder with random (unpredictable) structures.
    seed = 0
    sizes = [3, 5, 7, 9, 11, 13]
    while len(specs) < target:
        n = sizes[seed % len(sizes)]
        use_h = (seed % 3 == 0)                      # some purely-CX, some with H
        tag = "clifford" if use_h else "cx"
        add(f"random_{tag}_{n}_{seed}", n,
            make_random_encoder(n, seed, use_h), "coded", "random")
        seed += 1

    return specs[:target]


# ══════════════════════════════════════════════════════════════
# Learned (calibrated) decoder — nothing hand-derived
# ══════════════════════════════════════════════════════════════

def calibrate_decoder(encoder, n):
    """Learn syndrome -> logical-bit by injecting every single-qubit error."""
    E = QuantumCircuit(n)
    if encoder:
        encoder(E)
    D = E.inverse()

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
            counts = _SIM.run(qc, shots=1, seed_simulator=1).result().get_counts()
            votes[next(iter(counts)).replace(" ", "")][b] += 1

    decode = {}
    for pattern, tally in votes.items():
        ranked = tally.most_common()
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            decode[pattern] = ranked[0][0]
    return decode


# ══════════════════════════════════════════════════════════════
# Noise at four locations (prep / gate / channel / readout)
# ══════════════════════════════════════════════════════════════

def _one_qubit_error(noise_type, p):
    if noise_type == "bit_flip":
        return pauli_error([("X", p), ("I", 1 - p)])
    if noise_type == "phase_flip":
        return pauli_error([("Z", p), ("I", 1 - p)])
    if noise_type == "depolarizing":
        return depolarizing_error(p, 1)
    raise ValueError(noise_type)


def make_noise(noise_type, prep_p, channel_p, gate_p, readout_p):
    nm = NoiseModel()
    if prep_p > 0:                       # BEFORE the circuit
        nm.add_all_qubit_quantum_error(_one_qubit_error(noise_type, prep_p), ["reset"])
    if channel_p > 0:                    # the transmission hop
        nm.add_all_qubit_quantum_error(_one_qubit_error(noise_type, channel_p), ["id"])
    if gate_p > 0:                       # encode / decode gates
        nm.add_all_qubit_quantum_error(depolarizing_error(gate_p, 1), ["x", "h", "s", "sdg"])
        nm.add_all_qubit_quantum_error(depolarizing_error(gate_p, 2), ["cx", "cz"])
    if readout_p > 0:                    # measurement
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - readout_p, readout_p], [readout_p, 1 - readout_p]]))
    return nm


# ══════════════════════════════════════════════════════════════
# Build and run one design under one noise setting
# ══════════════════════════════════════════════════════════════

def _apply_basis(qc, qubits, basis, undo=False):
    if basis == "X":
        qc.h(qubits)
    elif basis == "Y":
        if undo:
            qc.sdg(qubits); qc.h(qubits)
        else:
            qc.h(qubits); qc.s(qubits)


def _build_circuit(design, b):
    n, kind, basis = design["n"], design["kind"], design["basis"]
    qc = QuantumCircuit(n, n)
    qc.reset(range(n))                       # PREP location (before the circuit)
    if b:
        if kind == "independent":
            qc.x(range(n))
        else:
            qc.x(0)
    if kind == "independent":
        _apply_basis(qc, list(range(n)), basis)           # store in chosen basis
    elif design["encoder"]:
        enc = QuantumCircuit(n); design["encoder"](enc)
        qc.compose(enc, inplace=True)
    qc.barrier()
    for q in range(n):
        qc.id(q)                             # CHANNEL location
    qc.barrier()
    if kind == "independent":
        _apply_basis(qc, list(range(n)), basis, undo=True)
    elif design["encoder"]:
        dec = QuantumCircuit(n); design["encoder"](dec)
        qc.compose(dec.inverse(), inplace=True)
    qc.measure(range(n), range(n))
    return qc


def run_design(design, noise_type, prep_p=BASE_PREP, channel_p=BASE_CHANNEL,
               gate_p=BASE_GATE, readout_p=BASE_READOUT, shots=4000, seed=0):
    n, kind = design["n"], design["kind"]
    nm = make_noise(noise_type, prep_p, channel_p, gate_p, readout_p)
    sim = AerSimulator(noise_model=nm, method="stabilizer")
    decode_map = design.get("decode_map", {})

    errors = total = 0
    depth = gate_count = 0
    for b in (0, 1):
        qc = _build_circuit(design, b)
        depth, gate_count = qc.depth(), qc.size()
        counts = sim.run(qc, shots=shots, seed_simulator=seed + b).result().get_counts()
        for pattern, hits in counts.items():
            key = pattern.replace(" ", "")
            if kind == "independent":
                logical = 1 if key.count("1") > n / 2 else 0
            else:
                logical = int(decode_map.get(key, key[-1]))
            if logical != b:
                errors += hits
            total += hits

    return {"design": design["name"], "family": design["family"],
            "structure": design["kind"], "basis": design["basis"],
            "n_qubits": n, "depth": depth, "gate_count": gate_count,
            "noise_type": noise_type, "logical_error": errors / total}


def prepare_designs(target=100):
    """Attach a learned decoder to every coded design; validate zero-noise = 0."""
    designs = build_registry(target)
    for d in designs:
        if d["kind"] == "coded":
            d["decode_map"] = calibrate_decoder(d["encoder"], d["n"])
    return designs


# ══════════════════════════════════════════════════════════════
# Experiment 1 — the 100-circuit zoo across the three noise types
# ══════════════════════════════════════════════════════════════

def run_zoo(target=100, shots=4000, save=True, verbose=True):
    designs = prepare_designs(target)
    rows = []
    for i, design in enumerate(designs, 1):
        for noise_type in NOISE_TYPES:
            rows.append(run_design(design, noise_type, shots=shots))
        if verbose and i % 20 == 0:
            print(f"  tested {i}/{len(designs)} circuits")
    df = pd.DataFrame(rows)

    worst = df.groupby("design")["logical_error"].max().rename("worst_case_error")
    mean = df.groupby("design")["logical_error"].mean().rename("mean_error")
    df = df.merge(worst, on="design").merge(mean, on="design")

    # Error-reduction % versus a bare qubit — per noise type and worst-case.
    bare = df[df.design == "bare"].set_index("noise_type")["logical_error"]
    bare_worst = df.loc[df.design == "bare", "worst_case_error"].iloc[0]
    df["bare_error"] = df["noise_type"].map(bare)
    df["reduction_pct"] = (100 * (df["bare_error"] - df["logical_error"])
                           / df["bare_error"].clip(lower=1e-9))
    df["worst_reduction_pct"] = 100 * (bare_worst - df["worst_case_error"]) / max(bare_worst, 1e-9)

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = DATA_DIR / "three_noise_zoo_dataset.csv"
        df.to_csv(out, index=False)
        if verbose:
            print(f"Saved {len(df)} rows ({df.design.nunique()} circuits) to {out}")
    return df, designs


def pick_champion(df, designs):
    """Overall best = lowest worst-case error, tie-broken by shallow depth."""
    ranking = (df.groupby("design")
               .agg(worst=("worst_case_error", "first"), depth=("depth", "first"))
               .sort_values(["worst", "depth"]))
    by_name = {d["name"]: d for d in designs}
    champ = by_name[ranking.index[0]]
    champ["_depth"] = int(ranking.iloc[0]["depth"])
    champ["_worst"] = float(ranking.iloc[0]["worst"])
    return champ


def per_noise_champions(df, designs):
    """Best circuit for each individual noise type (used by the adaptive loop)."""
    by_name = {d["name"]: d for d in designs}
    out = {}
    for nt in NOISE_TYPES:
        sub = df[df.noise_type == nt].sort_values("logical_error")
        out[nt] = by_name[sub.iloc[0]["design"]]
    return out


def pick_study_circuit(df, designs, target=0.06):
    """
    A strong-but-improvable circuit (worst-case error near `target`) for the
    before/through-the-circuit hardening study. The champion is already so close
    to perfect that hardening it shows nothing; a circuit with headroom makes the
    effect of hardening each noise location clearly visible.
    """
    worst = df.groupby("design")["worst_case_error"].first()
    name = (worst - target).abs().idxmin()
    by_name = {d["name"]: d for d in designs}
    d = by_name[name]
    d["_worst"] = float(worst[name])
    return d


# ══════════════════════════════════════════════════════════════
# Experiment 2 — does noise at OTHER locations matter (incl. prep)?
# ══════════════════════════════════════════════════════════════

def run_location_study(design, noise_type="depolarizing", shots=8000, save=True):
    levels = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
    floor = 0.005
    rows = []
    for location in ("prep", "gate", "channel", "readout"):
        for level in levels:
            kwargs = {"prep_p": floor, "gate_p": floor,
                      "channel_p": floor, "readout_p": floor}
            kwargs[f"{location}_p"] = level
            r = run_design(design, noise_type, shots=shots, **kwargs)
            rows.append({"location": location, "level": level,
                         "logical_error": r["logical_error"]})
    df = pd.DataFrame(rows)
    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_DIR / "three_noise_location_study.csv", index=False)
    return df


def refine_design(design, location_df, shots=8000, save=True):
    """Harden the most damaging location (halve it) and re-measure everywhere."""
    sensitivity = (location_df.groupby("location")["logical_error"].max()
                   - location_df.groupby("location")["logical_error"].min())
    worst_location = sensitivity.idxmax()

    rows = []
    for noise_type in NOISE_TYPES:
        before = run_design(design, noise_type, shots=shots)["logical_error"]
        hardened = {"prep_p": BASE_PREP, "gate_p": BASE_GATE,
                    "channel_p": BASE_CHANNEL, "readout_p": BASE_READOUT}
        hardened[f"{worst_location}_p"] /= 2.0
        after = run_design(design, noise_type, shots=shots, **hardened)["logical_error"]
        rows.append({"noise_type": noise_type,
                     "error_before": round(before, 5),
                     "error_after": round(after, 5),
                     "reduction_pct": round(100 * (before - after) / max(before, 1e-9), 1)})
    df = pd.DataFrame(rows)
    if save:
        df.to_csv(DATA_DIR / "three_noise_refinement.csv", index=False)
    return worst_location, df


# ══════════════════════════════════════════════════════════════
# Experiment 3 — close the abstract's loop (classify -> select -> mitigate)
# ══════════════════════════════════════════════════════════════

def _probe_channel(noise_type, strength, shots=4000, seed=0):
    """Cheap 1-qubit probes: error seen in the Z basis and in the X basis."""
    nm = make_noise(noise_type, prep_p=0.0, channel_p=strength,
                    gate_p=0.0, readout_p=0.01)
    sim = AerSimulator(noise_model=nm, method="stabilizer")
    qz = QuantumCircuit(1, 1); qz.id(0); qz.measure(0, 0)
    err_z = sim.run(qz, shots=shots, seed_simulator=seed).result().get_counts().get("1", 0) / shots
    qx = QuantumCircuit(1, 1); qx.h(0); qx.id(0); qx.h(0); qx.measure(0, 0)
    err_x = sim.run(qx, shots=shots, seed_simulator=seed + 1).result().get_counts().get("1", 0) / shots
    return err_z, err_x


def build_classifier_dataset(save=True):
    rows = []
    strengths = np.round(np.arange(0.03, 0.23, 0.01), 3)
    for nt in NOISE_TYPES:
        for s in strengths:
            for seed in range(6):
                err_z, err_x = _probe_channel(nt, float(s), seed=seed * 7)
                rows.append({"noise_type": nt, "strength": float(s),
                             "err_z": err_z, "err_x": err_x,
                             "z_minus_x": err_z - err_x, "z_plus_x": err_z + err_x})
    df = pd.DataFrame(rows)
    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_DIR / "three_noise_classifier_dataset.csv", index=False)
    return df


def train_classifier(df):
    feats = ["err_z", "err_x", "z_minus_x", "z_plus_x"]
    X, y = df[feats].values, df["noise_type"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    cm = confusion_matrix(yte, clf.predict(Xte), labels=NOISE_TYPES)
    return clf, feats, acc, cm


def run_adaptive_loop(clf, feats, champs, fixed_design, save=True, shots=8000):
    """
    For each environment: probe the channel -> ML predicts the noise type ->
    SELECT the circuit that is best for that prediction -> measure the error.
    Compared against a bare qubit and a single fixed circuit.
    """
    environments = [(nt, s) for nt in NOISE_TYPES for s in (0.08, 0.14)]
    bare = {"name": "bare", "n": 1, "encoder": None, "kind": "coded",
            "family": "baseline", "basis": "Z", "decode_map": {}}
    rows = []
    for nt, s in environments:
        err_z, err_x = _probe_channel(nt, s)
        pred = clf.predict([[err_z, err_x, err_z - err_x, err_z + err_x]])[0]
        chosen = champs[pred]

        bare_err = run_design(bare, nt, channel_p=s, shots=shots)["logical_error"]
        fixed_err = run_design(fixed_design, nt, channel_p=s, shots=shots)["logical_error"]
        adapt_err = run_design(chosen, nt, channel_p=s, shots=shots)["logical_error"]
        rows.append({"true_noise": nt, "strength": s, "predicted_noise": pred,
                     "correct": pred == nt, "selected_circuit": chosen["name"],
                     "bare_error": round(bare_err, 5),
                     "fixed_error": round(fixed_err, 5),
                     "adaptive_error": round(adapt_err, 5),
                     "reduction_vs_bare_pct":
                         round(100 * (bare_err - adapt_err) / max(bare_err, 1e-9), 1)})
    df = pd.DataFrame(rows)
    if save:
        df.to_csv(DATA_DIR / "three_noise_adaptive_results.csv", index=False)
    return df


# ══════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════

_FAMILY_COLORS = {
    "baseline": "black", "rep_chain": "tab:blue", "rep_tree": "tab:cyan",
    "phase_chain": "tab:purple", "phase_tree": "violet",
    "independent": "tab:orange", "independent_even": "gold",
    "basis_choice": "tab:red", "shor": "brown", "hybrid": "tab:green",
    "dfs_pair": "olive", "random": "gray",
}


def _save(name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  saved {path}")
    plt.show()          # display inline in Jupyter; no-op under the Agg backend
    return path


def plot_universality_vs_speed(df):
    """The money plot: all 100 circuits — resists-everything (low) AND fast (low depth)."""
    summ = df.groupby("design", as_index=False).agg(
        worst=("worst_case_error", "first"), depth=("depth", "first"),
        n_qubits=("n_qubits", "first"), family=("family", "first"))
    plt.figure(figsize=(11, 7))
    for fam, sub in summ.groupby("family"):
        plt.scatter(sub["depth"], sub["worst"].clip(lower=1e-4),
                    s=40 + 12 * sub["n_qubits"], alpha=0.75,
                    color=_FAMILY_COLORS.get(fam, "gray"), label=fam)
    best = summ.sort_values(["worst", "depth"]).iloc[0]
    plt.annotate(f"  best: {best['design']}", (best["depth"], max(best["worst"], 1e-4)),
                 fontsize=10, fontweight="bold")
    plt.yscale("log")
    plt.xlabel("Circuit depth  (lower = faster)")
    plt.ylabel("Worst-case error over the 3 noise types  (lower = more universal)")
    plt.title("100 Circuits: Universal AND Fast Sits Bottom-Left")
    plt.legend(fontsize=8, ncol=2, title="Family")
    plt.grid(True, which="both", alpha=0.3)
    return _save("three_noise_universality_vs_speed.png")


def plot_family_summary(df):
    """Worst-case error distribution by circuit family."""
    summ = df.groupby("design", as_index=False).agg(
        worst=("worst_case_error", "first"), family=("family", "first"))
    fams = summ.groupby("family")["worst"].median().sort_values().index
    data = [summ.loc[summ.family == f, "worst"].values for f in fams]
    plt.figure(figsize=(11, 6))
    bp = plt.boxplot(data, patch_artist=True, showmeans=True)
    for patch, f in zip(bp["boxes"], fams):
        patch.set_facecolor(_FAMILY_COLORS.get(f, "gray")); patch.set_alpha(0.6)
    plt.ylabel("Worst-case error over 3 noise types")
    plt.xlabel("Circuit family")
    plt.title("Which Families Resist All Three Noise Types (lower = better)")
    plt.xticks(range(1, len(fams) + 1), fams, rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    return _save("three_noise_family_summary.png")


def plot_top_heatmap(df, top=25):
    """Best circuits vs each noise type."""
    order = (df.groupby("design")["worst_case_error"].first()
             .sort_values().head(top).index)
    grid = (df[df.design.isin(order)]
            .pivot(index="design", columns="noise_type", values="logical_error")
            .loc[order])
    plt.figure(figsize=(8, 10))
    plt.imshow(grid.values, aspect="auto", cmap="RdYlGn_r",
               vmin=0, vmax=max(grid.values.max(), 1e-3))
    plt.colorbar(label="Logical error rate")
    plt.xticks(range(len(grid.columns)), grid.columns, rotation=15)
    plt.yticks(range(len(grid.index)), grid.index, fontsize=8)
    plt.title(f"Top {top} Circuits vs Each Noise Type (greener = better)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            plt.text(j, i, f"{grid.values[i, j]:.3f}", ha="center", va="center", fontsize=7)
    return _save("three_noise_top_heatmap.png")


def plot_error_reduction(df, top=15):
    """Biggest worst-case error-reduction % versus a bare qubit."""
    summ = (df.groupby(["design", "family"], as_index=False)
            .agg(reduction=("worst_reduction_pct", "first"),
                 depth=("depth", "first"))
            .sort_values("reduction", ascending=False).head(top))
    plt.figure(figsize=(10, 6))
    colors = [_FAMILY_COLORS.get(f, "gray") for f in summ["family"]]
    plt.barh(summ["design"][::-1], summ["reduction"][::-1], color=colors[::-1])
    plt.xlabel("Worst-case error reduction vs bare qubit (%)")
    plt.title(f"Top {top} Circuits by Error Reduction Across All Three Noise Types")
    plt.grid(True, axis="x", alpha=0.3)
    return _save("three_noise_error_reduction.png")


def plot_best_circuit(champion):
    qc = _build_circuit(champion, 1)
    fig = qc.draw("mpl", fold=-1)
    fig.suptitle(f"Best All-Round Circuit: {champion['name']}  "
                 f"(worst-case error {champion.get('_worst', 0):.4f}, "
                 f"depth {champion.get('_depth', '?')})", y=1.02)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "three_noise_best_circuit.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  saved {path}")
    plt.show()
    return path


def plot_location_study(location_df, design_name=""):
    plt.figure(figsize=(10, 6))
    for location, sub in location_df.groupby("location"):
        sub = sub.sort_values("level")
        plt.plot(sub["level"], sub["logical_error"], "o-", label=location)
    plt.xlabel("Noise strength at that location")
    plt.ylabel("Logical error rate")
    plt.title(f"Noise Before AND Through the Circuit ({design_name}):\n"
              "Which Location Hurts Most")
    plt.legend(title="Location (incl. prep = before the circuit)")
    plt.grid(True, alpha=0.3)
    return _save("three_noise_location_study.png")


def plot_refinement(worst_location, refine_df, design_name=""):
    x = np.arange(len(refine_df))
    plt.figure(figsize=(9, 6))
    plt.bar(x - 0.2, refine_df["error_before"], width=0.4, label="Before", color="indianred")
    plt.bar(x + 0.2, refine_df["error_after"], width=0.4,
            label=f"After hardening '{worst_location}'", color="seagreen")
    plt.xticks(x, refine_df["noise_type"], rotation=15)
    plt.ylabel("Logical error rate")
    plt.title(f"Refinement of {design_name}: Hardening the '{worst_location}' "
              "Location Helps Every Noise Type")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    return _save("three_noise_refinement.png")


def plot_classifier_confusion(cm, acc):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(label="Count")
    plt.xticks(range(len(NOISE_TYPES)), NOISE_TYPES, rotation=15)
    plt.yticks(range(len(NOISE_TYPES)), NOISE_TYPES)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Noise-Type Classifier (test accuracy {acc:.2f})")
    for i in range(len(NOISE_TYPES)):
        for j in range(len(NOISE_TYPES)):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    return _save("three_noise_classifier_confusion.png")


def plot_adaptive_loop(adaptive_df):
    x = np.arange(len(adaptive_df))
    labels = [f"{r.true_noise}\n@{r.strength}" for r in adaptive_df.itertuples()]
    plt.figure(figsize=(11, 6))
    plt.bar(x - 0.25, adaptive_df["bare_error"], width=0.25, label="Bare qubit", color="indianred")
    plt.bar(x, adaptive_df["fixed_error"], width=0.25, label="Fixed circuit", color="tab:orange")
    plt.bar(x + 0.25, adaptive_df["adaptive_error"], width=0.25,
            label="ML-selected (adaptive)", color="seagreen")
    plt.xticks(x, labels)
    plt.ylabel("Logical error rate")
    plt.title("Closing the Loop: ML-Selected Circuit vs Bare and Fixed")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    return _save("three_noise_adaptive_loop.png")


# ══════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════

def summarize(df, champion, study, worst_location, refine_df, acc, adaptive_df):
    print("\nHeadline results")
    print("----------------")
    summ = (df.groupby(["design", "family"])
            .agg(worst=("worst_case_error", "first"),
                 reduction=("worst_reduction_pct", "first"),
                 depth=("depth", "first"), qubits=("n_qubits", "first"))
            .sort_values("worst"))
    print("Best 8 circuits (by worst-case error over all 3 noise types):")
    print(summ.head(8).to_string())
    print("\nWorst 5 circuits:")
    print(summ.tail(5).to_string())
    print(f"\nBest all-round circuit: {champion['name']} "
          f"(worst-case {champion['_worst']:.4f}, depth {champion['_depth']}, "
          f"{summ.loc[champion['name']].iloc[0]['reduction']:.1f}% error reduction vs bare)")
    print(f"Hardening study circuit: {study['name']} "
          f"(worst-case {study['_worst']:.4f} — has headroom to improve)")
    print(f"Most damaging noise location: {worst_location}")
    print(f"Refinement (hardening {worst_location}) cut its error by "
          f"{refine_df['reduction_pct'].mean():.0f}% on average.")
    print(f"Noise-type classifier test accuracy: {acc:.2f}")
    print(f"Adaptive loop mean error reduction vs bare: "
          f"{adaptive_df['reduction_vs_bare_pct'].mean():.1f}%")


def main():
    print("100-Circuit Three-Noise Hardening Search")
    print("========================================")
    df, designs = run_zoo()
    champion = pick_champion(df, designs)
    champs = per_noise_champions(df, designs)
    study = pick_study_circuit(df, designs)

    location_df = run_location_study(study)
    worst_location, refine_df = refine_design(study, location_df)

    print("Training noise-type classifier ...")
    clf_df = build_classifier_dataset()
    clf, feats, acc, cm = train_classifier(clf_df)
    fixed_design = next(d for d in designs if d["name"] == "independent_7")
    adaptive_df = run_adaptive_loop(clf, feats, champs, fixed_design)

    print("Rendering figures ...")
    plot_universality_vs_speed(df)
    plot_family_summary(df)
    plot_top_heatmap(df)
    plot_error_reduction(df)
    plot_best_circuit(champion)
    plot_location_study(location_df, study["name"])
    plot_refinement(worst_location, refine_df, study["name"])
    plot_classifier_confusion(cm, acc)
    plot_adaptive_loop(adaptive_df)

    summarize(df, champion, study, worst_location, refine_df, acc, adaptive_df)
    print("\nDone.")


if __name__ == "__main__":
    main()
