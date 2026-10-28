# ML DevOps ISE2: MLflow and Data Drift Analysis

## Overview
This project demonstrates ML experiment tracking using MLflow and data drift detection using statistical testing.

## Technologies Used
- Python
- MLflow
- Scikit-learn
- Pandas
- SciPy
- Matplotlib
- Pytest

## Experiment 1: MLflow Experiment Tracking
A Random Forest classifier is trained using the Scikit-learn Breast Cancer dataset. MLflow tracks experiment parameters, evaluation metrics, and the trained model.

Run:
python mlflow_experiment.py

## Experiment 2: Data Drift Detection
The drift analysis compares reference and modified datasets using the two-sample Kolmogorov-Smirnov (KS) test. A p-value below 0.05 is treated as evidence of drift. The current dataset is deliberately modified for demonstration purposes.

Run:
python drift_analysis.py

## Output Files
- reports/drift_report.csv — statistical drift results
- reports/drift_comparison.png — visual comparison of feature distributions

## MLflow UI
Run:
mlflow ui

Open the local URL printed in the terminal to view experiments, runs, metrics, and artifacts.

## Setup
Install dependencies using:
python -m pip install -r requirements.txt
