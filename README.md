# Adaptive Machine Learning for BB84 Quantum Key Distribution

Machine Learning-Assisted Noise Analysis for BB84 Quantum Key Distribution in Noisy Quantum Channels

## Abstract

Quantum Key Distribution (QKD) offers a fundamentally secure method for cryptographic key exchange by using the principles of quantum mechanics. Among existing QKD protocols, BB84 remains one of the most widely studied and implemented approaches. However, the performance and reliability of BB84 can be degraded by quantum-channel and measurement noise, resulting in increased Quantum Bit Error Rates (QBER) and reduced key-generation reliability.

This project develops a Qiskit-based BB84 simulation framework that models communication under multiple channel conditions, including ideal transmission, depolarizing noise, readout noise, amplitude-damping noise, and a combined-noise configuration. The simulation can also include an intercept-resend eavesdropper, Eve, so that the effects of both channel noise and eavesdropping can be compared.

Hundreds of complete BB84 sessions are run across different noise strengths and channel conditions to generate a structured CSV dataset. Session-level measurements include QBER, key accuracy, sifted-key length, basis-specific QBER, directional error rates, and related channel statistics.

A Random Forest classifier is trained to classify the observed channel condition as ideal, low noise, medium noise, or high noise. The model is evaluated using accuracy, balanced accuracy, macro F1-score, per-class precision and recall, a classification report, a confusion matrix, and feature-importance analysis.

The current implementation focuses on BB84 simulation, noise characterization, dataset generation, and noise-severity prediction. Adaptive mitigation selection is a planned extension and is not yet implemented in the current codebase.

## Research Problem

Quantum Key Distribution provides theoretically secure key exchange, but physical noise and measurement errors can significantly degrade communication performance.

Traditional BB84 systems identify errors after transmission by comparing part of the sifted key and calculating QBER. This project investigates whether machine learning can learn from BB84 session statistics and classify noisy channel conditions accurately enough to support future adaptive mitigation strategies.

### Research Question

Can a machine-learning model identify BB84 channel-noise severity from simulation measurements and provide a foundation for selecting appropriate mitigation strategies?

## Project Objectives

- Implement BB84 Quantum Key Distribution using Qiskit Aer.
- Simulate Alice, Bob, and an optional intercept-resend eavesdropper.
- Implement multiple controlled noise models.
- Measure QBER and key-generation performance.
- Run hundreds of BB84 simulation sessions.
- Generate a reusable CSV dataset.
- Train a Random Forest noise-severity classifier.
- Evaluate the classifier using standard prediction metrics.
- Prepare the project for future adaptive noise-mitigation research.

## Current Project Schedule

### Week 2 — BB84 in Qiskit

Implemented:

- Alice's random bits and basis choices
- Bob's random basis choices
- Optional Eve intercept-resend attack
- Qiskit Aer simulation
- Basis comparison and key sifting
- Sifted-key length
- Key accuracy
- Quantum Bit Error Rate

### Week 3 — Noise Models and Graphs

Implemented:

- Ideal baseline
- Depolarizing noise
- Readout noise
- Amplitude-damping noise
- Combined noise
- QBER versus noise-strength graph
- Eve-present versus Eve-absent QBER graph
- Directional amplitude-damping error graph

Generated figures are saved in:

```text
results/figures/
```

### Week 4 — Dataset Generation

The default full experiment runs 220 complete BB84 sessions:

- 20 ideal sessions
- 200 noisy sessions
- 4 noisy channel configurations
- 5 nonzero noise strengths
- Eve present and absent
- 128 transmitted qubits per session

The generated dataset is saved as:

```text
data/processed/bb84_experiment_dataset.csv
```

Each row represents one complete BB84 session rather than one individual qubit.

### Week 5 — Random Forest Model

The Random Forest classifier predicts:

- `ideal`
- `low`
- `medium`
- `high`

The model does not receive the true noise strength, noise type, noise class, run ID, or repetition number as input features.

Generated outputs include:

- Accuracy
- Balanced accuracy
- Macro F1-score
- Precision and recall by class
- Classification report
- Confusion matrix
- Feature importance
- Saved Joblib model

Results are saved in:

```text
results/metrics/
results/models/
results/figures/
```

## Repository Structure

```text
adaptive-machine-learning/
├── .gitignore
├── .venv/                         # Local only; not committed to GitHub
├── .vscode/
│   └── settings.json
├── notebooks/
│   └── BB84_Complete_Weeks_2_5.ipynb
├── src/
│   └── bb84/
│       ├── __init__.py
│       ├── core.py
│       └── pipeline.py
├── data/
│   └── processed/
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── run_pipeline.py
├── requirements.txt
└── README.md
```

## Prerequisites

Install the following before setting up the project:

- Git
- Python 3.10 or newer
- WSL with Ubuntu, Linux, or another Python-compatible environment
- Visual Studio Code
- VS Code Python extension
- VS Code Jupyter extension
- VS Code WSL extension when using WSL

## Clone and Set Up the Project

Each teammate must create their own local Python virtual environment. The `.venv` directory is intentionally excluded from GitHub because virtual environments contain machine-specific paths and installed binaries.

### 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd adaptive-machine-learning
```

Replace `<YOUR-REPOSITORY-URL>` with the actual GitHub repository URL.

### 2. Create and configure the virtual environment

For WSL or Linux, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name adaptive-machine-learning --display-name "Python (adaptive-machine-learning)"
```

The same setup can be completed in one command:

```bash
python3 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip && python -m pip install -r requirements.txt && python -m ipykernel install --user --name adaptive-machine-learning --display-name "Python (adaptive-machine-learning)"
```

### 3. Activate the environment during future sessions

The environment only needs to be created once. Each time the project is reopened, activate it with:

```bash
source .venv/bin/activate
```

To leave the virtual environment:

```bash
deactivate
```

## Run a Quick Validation

Run a smaller experiment first to confirm that the environment and dependencies work:

```bash
python run_pipeline.py --quick
```

The quick run creates a reduced dataset and verifies the complete simulation and machine-learning pipeline.

## Run the Full Experiment

Run the complete Week 2-5 pipeline with:

```bash
python run_pipeline.py
```

The program will:

1. Run the BB84 simulations.
2. Apply the configured noise models.
3. Generate the experiment CSV.
4. Create the noise-effect graphs.
5. Train the Random Forest classifier.
6. Save prediction metrics and model outputs.

The full run can take several minutes.

## Run the Jupyter Notebook

Open the project from WSL:

```bash
code .
```

Then open:

```text
notebooks/BB84_Complete_Weeks_2_5.ipynb
```

In the upper-right corner of the notebook:

1. Click **Select Kernel**.
2. Select **Python Environments** or **Jupyter Kernel**.
3. Choose **Python (adaptive-machine-learning)** or the interpreter at `.venv/bin/python`.
4. Click **Run All**.

## Main Output Files

### Dataset

```text
data/processed/bb84_experiment_dataset.csv
```

### Noise graphs

```text
results/figures/qber_vs_noise.png
results/figures/qber_eve_comparison.png
results/figures/amplitude_damping_directionality.png
```

### Machine-learning graphs

```text
results/figures/random_forest_confusion_matrix.png
results/figures/random_forest_feature_importance.png
```

### Prediction metrics

```text
results/metrics/random_forest_metrics.json
results/metrics/classification_report.csv
results/metrics/feature_importance.csv
```

### Trained model

```text
results/models/bb84_noise_severity_random_forest.joblib
```

## Git and Virtual Environment Rules

The local `.venv` folder should not be committed to GitHub.

The `.gitignore` file should include:

```gitignore
.venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
```

The repository should include `requirements.txt`, which allows every teammate to install the same required Python packages in their own environment.

Files that should be committed include:

```text
.gitignore
README.md
requirements.txt
run_pipeline.py
notebooks/
src/
```

Files and folders that should not be committed include:

```text
.venv/
__pycache__/
.ipynb_checkpoints/
```

## Team Git Workflow

Before beginning work:

```bash
git pull
```

Create a branch for new work:

```bash
git checkout -b feature/descriptive-branch-name
```

After making changes:

```bash
git add .
git commit -m "Describe the completed change"
git push -u origin feature/descriptive-branch-name
```

Then open a pull request on GitHub so the changes can be reviewed before being merged into the main branch.

## Reproducibility

The project uses fixed random seeds where appropriate so that simulation and machine-learning results can be reproduced more consistently. Exact runtime and some numerical results can still vary slightly depending on the operating system, Python version, and installed dependency versions.

## Current Limitations

- The current code classifies noise severity but does not yet apply an adaptive mitigation action.
- The simulation uses Qiskit Aer rather than physical quantum-network hardware.
- The default dataset contains 220 sessions and may be expanded for stronger machine-learning evaluation.
- Additional models and validation strategies may be needed before making claims about real-world QKD performance.
- The current implemented noise configurations are ideal, depolarizing, readout, amplitude damping, and combined noise.

## Planned Extensions

Possible future work includes:

- Adaptive mitigation selection based on the predicted channel class
- Bit-flip and phase-flip channel models
- Additional machine-learning algorithms
- Hyperparameter optimization
- Cross-validation
- Larger datasets
- Real-device or hardware-calibrated noise models
- Statistical comparison between mitigation strategies
- Secure-key-rate estimation
- Integration with research-paper retrieval tools as a separate supporting system

## License

Add the project's selected license information here.
