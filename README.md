# Adaptive Machine Learning for BB84 Quantum Key Distribution

Adaptive Machine Learning-Assisted Noise Mitigation for BB84 Quantum Key Distribution in Noisy Quantum Channels

## Abstract

Quantum Key Distribution (QKD) offers a fundamentally secure method for cryptographic key exchange using the principles of quantum mechanics. Among existing protocols, BB84 remains one of the most widely studied. However, its performance is degraded by quantum-channel and measurement noise, which raises the Quantum Bit Error Rate (QBER) and reduces key-generation reliability.

This project builds a Qiskit-based BB84 simulation framework, models four canonical noise types (**bit-flip, phase-flip, depolarizing, and amplitude-damping**), generates a structured dataset relating channel behavior to QBER and key-generation performance, and trains a machine-learning classifier to identify the noise condition from observable session statistics. Crucially, the framework then **closes the loop**: based on the predicted channel state it dynamically selects a mitigation action and measures the resulting improvement. Separate studies quantify how much the protocol can be hardened against noise and search for unusual circuit structures that resist every noise type while staying fast.

## Research Problem

QKD is theoretically secure, but physical noise degrades performance, and traditional BB84 only detects errors *after* transmission by comparing part of the sifted key. This project investigates whether machine learning can identify noisy channel conditions from BB84 session statistics and **select an appropriate mitigation strategy** to improve performance.

**Research question:** Can an ML model classify BB84 channel-noise conditions from simulation measurements and drive the selection of mitigation strategies that measurably improve the protocol?

## What This Project Delivers

The classify → select → mitigate loop is implemented, not planned:

1. **Simulate** BB84 under four noise types at varying strengths (with an optional intercept-resend Eve).
2. **Generate** a CSV dataset of session statistics (QBER, basis-specific QBER, sifted-key rate, directional bias, and more).
3. **Classify** the noise type with a Random Forest (~0.83 cross-validated accuracy).
4. **Select and apply** a mitigation action based on the prediction, and compare QBER before and after.
5. **Harden** the protocol with quantum error-correcting codes and quantify the speed/cost of doing so.
6. **Search** for circuit structures that resist all noise types at once while remaining fast.

## Repository Structure

```text
adaptive-machine-learning/
├── .gitignore
├── .venv/                              # Local only; not committed
├── .vscode/
│   └── settings.json
├── BB84_Adaptive_QKD.py                # Core module: Weeks 2-6 (simulate, dataset, classify, mitigate)
├── noise_resistance.py                 # Circuit-hardening study + ML-selected mitigation capstone
├── advanced_circuits.py                # Separate test: advanced circuit search, multi-location noise
├── notebooks/
│   └── BB84_Adaptive_QKD.ipynb         # Runs and documents everything end to end
├── data/
│   └── processed/                      # Generated datasets (CSV)
├── results/
│   └── figures/                        # Generated figures (PNG)
├── requirements.txt
└── README.md
```

All three `.py` files are importable modules (functions only, guarded by `if __name__ == "__main__"`). The notebook imports them to produce the figures and datasets for documentation.

## The Three Modules

### 1. `BB84_Adaptive_QKD.py` — Core protocol, dataset, classifier, mitigation (Weeks 2-6)

- **Week 2:** BB84 in Qiskit Aer — Alice's bits/bases, Bob's bases, optional intercept-resend Eve, sifting, QBER.
- **Week 3:** Four noise models — `bit_flip`, `phase_flip`, `depolarizing`, `amplitude_damping` — across a range of strengths.
- **Week 4:** Dataset generation. Each row is one full BB84 session with derived features, including `basis_qber_diff` (separates bit-flip vs phase-flip) and `bit_bias` (an amplitude-damping fingerprint).
- **Week 5:** A Random Forest classifies the **noise type** (~0.83 cross-validated accuracy).
- **Week 6:** Adaptive mitigation — the predicted noise type selects a mitigation action, and the demo table reports predicted vs true type.

Performance note: sessions are simulated by precomputing the eight possible `(bit, prep-basis, measure-basis)` measurement probabilities once, then Bernoulli-sampling all qubits with NumPy. A full 1000-session dataset generates in roughly 30 seconds.

Output: `data/processed/bb84_dataset.csv`

### 2. `noise_resistance.py` — Circuit hardening + ML-driven mitigation capstone

Studies how much BB84 can be strengthened against noise using **repetition codes** (bit-flip and phase-flip) at code distances 1–15, across noise strengths — 62 circuits in total. The logical error rate falls exponentially with code distance (best case ≈ 273,000× improvement: 5% physical → ~1.8e-7 logical at distance 15), while circuit depth and qubit count grow linearly — the measured cost of hardening.

It also contains the capstone `adaptive_mitigation` routine that ties the loop together: run a BB84 session → the ML model predicts the noise type → the matching error-correcting code is selected → the protected QBER is measured.

Outputs: `data/processed/noise_resistance_dataset.csv`, plus `results/figures/hardening_*.png` and `adaptive_ml_selected_mitigation.png`.

### 3. `advanced_circuits.py` — Separate test: advanced circuit search

A standalone exploration (imports nothing from the other modules) that searches for **unpredictable** circuit structures resistant to *all* noise types while staying fast. It builds a circuit zoo — bare qubit, chain vs tree repetition encoders, phase codes, independent re-transmission (send the bit N times with no entangling gates), and the Shor 9-qubit code — with **learned decoders** (each circuit calibrates its own syndrome→bit map by simulation) and injects noise at **three locations**: gate, channel, and readout.

Key finding (counterintuitive): the champion is **independent re-transmission** (worst-case error ≈ 0.0025 across all four noise types at depth 3, as fast as a bare qubit), while the sophisticated Shor-9 code comes last — its entangling gates inject more noise than they correct. The location study shows readout noise dominates; hardening it cut error a further ~20%.

Outputs: `data/processed/advanced_circuits_dataset.csv`, `data/processed/advanced_location_study.csv`, plus `results/figures/advanced_*.png`.

## Prerequisites

- Git
- Python 3.10 or newer
- WSL (Ubuntu/Linux) or another Python-compatible environment
- Visual Studio Code with the Python, Jupyter, and (for WSL) WSL extensions

## Set Up the Project

The `.venv` directory is intentionally excluded from Git because virtual environments contain machine-specific paths. Each teammate creates their own.

### 1. Clone

```bash
git clone <YOUR-REPOSITORY-URL>
cd adaptive-machine-learning
```

### 2. Create and configure the virtual environment (WSL / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name adaptive-machine-learning --display-name "Python (adaptive-machine-learning)"
```

### 3. Reactivate in future sessions

```bash
source .venv/bin/activate    # deactivate to leave
```

## Run the Code

Each module can be run directly to regenerate its datasets and figures:

```bash
source .venv/bin/activate

python BB84_Adaptive_QKD.py     # dataset + classifier + adaptive-mitigation demo
python noise_resistance.py      # circuit-hardening study + ML-selected mitigation
python advanced_circuits.py     # separate test: advanced circuit search
```

> The virtual environment is a Linux/WSL environment. On Windows, run these through WSL (for example `wsl -e bash -c "cd <repo> && .venv/bin/python BB84_Adaptive_QKD.py"`); the Windows Python interpreter cannot run the Linux `.venv` directly. For headless figure generation, set `MPLBACKEND=Agg`.

## Run the Jupyter Notebook

The notebook imports all three modules and reproduces every dataset and figure with narrative documentation.

Open the project from WSL (`code .`), then open:

```text
notebooks/BB84_Adaptive_QKD.ipynb
```

Select the **Python (adaptive-machine-learning)** kernel (or the interpreter at `.venv/bin/python`) and choose **Run All**. Sections: Weeks 2–6, Circuit Hardening, Closing the Loop — ML-Selected Mitigation, Separate Test — Advanced Circuit Search, and Conclusion.

## Main Output Files

### Datasets

```text
data/processed/bb84_dataset.csv                # BB84 sessions across four noise types
data/processed/noise_resistance_dataset.csv    # Repetition-code hardening results
data/processed/advanced_circuits_dataset.csv   # Circuit-zoo results across noise types
data/processed/advanced_location_study.csv     # Gate/channel/readout noise-location study
```

### Figures

```text
results/figures/adaptive_ml_selected_mitigation.png    # The classify→select→mitigate loop
results/figures/hardening_*.png                        # Logical error vs distance, cost, trade-offs
results/figures/advanced_*.png                         # Circuit zoo, universality vs speed, champion
```

## Git and Virtual Environment Rules

Do not commit `.venv/`. The `.gitignore` covers:

```gitignore
.venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
```

Commit `requirements.txt` so teammates can reproduce the environment.

## Team Git Workflow

```bash
git pull
git checkout -b feature/descriptive-branch-name
# ...make changes...
git add .
git commit -m "Describe the completed change"
git push -u origin feature/descriptive-branch-name
```

Then open a pull request on GitHub for review before merging.

## Reproducibility

Fixed random seeds are used where appropriate so simulation and ML results reproduce consistently. Exact runtimes and some numerical values can still vary slightly across operating systems, Python versions, and dependency versions.

## Limitations

- Simulations use Qiskit Aer, not physical quantum hardware.
- ML classification accuracy is intentionally not the sole objective — the emphasis is on driving a genuine mitigation-selection loop and on studying circuit-level noise resistance.
- The advanced circuit search is a comparative simulation study; its findings motivate, but do not by themselves prove, real-world QKD performance.

## License

Add the project's selected license information here.
