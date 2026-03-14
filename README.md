# Temporal Attention-Guided Sequence Learning for Datacenter Network Failures

## Overview
This repository contains the official PyTorch implementation and dataset parsing logic for highly-generalizable, early-warning network anomaly prediction. As hyperscale topologies grow increasingly complex, classical machine learning algorithms struggle to map chaotic Layer-3 fault signatures. 

This project introduces a **Self-Attention Long Short-Term Memory (LSTM)** neural network. By utilizing a 15-step continuous sequence rolling window balanced natively with **SMOTE** (Synthetic Minority Oversampling Technique), the model successfully strips false-positive traffic spikes to predict hard network outages up to a full **60 seconds before** they physically manifest.

## Project Structure
- `src/attention_predictor.py`: The core PyTorch Neural Network architecture, Actionable 60-Second Window validation metrics, and Optical Hardware parsing logic.
- `src/cisco_predictor.py`: Adapter logic to map abstract BGP routing anomalies from the Cisco dataset into the Attention-LSTM.
- `src/merged_predictor.py`: The Omni-Hardware validation script. Uses mathematical array zero-padding to train a single model concurrently across both datasets.
- `src/generate_graphs.py`: Matplotlib scripts to output publication-quality F1 bar charts and Training Loss convergence curves.
- `results/`: Contains the generated high-resolution plots, raw `.md` metric logs, and the formal IEEE academic LaTeX paper draft (`paper.tex`).

## Datasets
Due to storage constraints, the raw datasets are not included in the repository and must be downloaded to the `telemetry/` folder prior to execution.
1. **Gigabit Optical Failure Dataset**: (Scuola Superiore Sant'Anna Testbed). Real-world metric tracking of physical light-loss.
2. **Cisco BGP Telemetry Dataset**: (Cisco Innovation Edge VIRL Topologies). Massively imbalanced 700,000-row tracker of abstract Layer-3 packet routing drops.

## Setup and Usage

### Prerequisites
- Python 3.10+
- PyTorch (CUDA supported for GPU acceleration)

### Installation
1. Clone the repository and navigate into the directory:
   ```bash
   git clone https://github.com/dheerajramasahayam/datacenter-network-failure-predictor.git
   cd datacenter-network-failure-predictor
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```
3. Install the required data-science dependencies:
   ```bash
   pip install torch pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
   ```

### Running the Architecture
You can test the zero-shot generalization of the Attention-LSTM across different networking hardware topographies:

1. **Verify Optical Hardware Performance**
   ```bash
   python src/attention_predictor.py
   ```
2. **Verify Cisco BGP Routing Performance**
   ```bash
   python src/cisco_predictor.py
   ```
3. **Verify the Zero-Padded Omni-Architecture**
   ```bash
   python src/merged_predictor.py
   ```
4. **Generate the Publication Visuals**
   ```bash
   python src/generate_graphs.py
   ```

## Academic Findings
The model yielded a maximal **0.46 F1-Score** detecting anomalies up to 60 seconds before total equipment blackout on the massively imbalanced Cisco testbed.
All hyperparameter tuning validations, methodology, and LaTeX derivations can be found in `results/paper.tex`.
