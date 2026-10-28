
import os
import mlflow
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. Set up MLflow
mlflow.set_experiment("ISE2_Data_Drift_Analysis")

# 2. Load the reference dataset
dataset = load_breast_cancer(as_frame=True)
data = dataset.data

reference, current = train_test_split(
    data, test_size=0.5, random_state=42
)

reference = reference.reset_index(drop=True)
current = current.reset_index(drop=True)

# 3. Simulate data drift in two features
features_to_shift = ["mean radius", "mean texture"]

for feature in features_to_shift:
    shift = 1.5 * reference[feature].std()
    current[feature] = current[feature] + shift

# 4. Detect drift using the Kolmogorov-Smirnov test
threshold = 0.05
results = []

for feature in data.columns:
    statistic, p_value = ks_2samp(
        reference[feature], current[feature]
    )

    drift_detected = p_value < threshold

    results.append({
        "feature": feature,
        "ks_statistic": statistic,
        "p_value": p_value,
        "drift_detected": drift_detected
    })

report = pd.DataFrame(results)

# 5. Create output folder and save the report
os.makedirs("reports", exist_ok=True)
report_path = "reports/drift_report.csv"
report.to_csv(report_path, index=False)

# 6. Plot reference vs current data
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, feature in zip(axes, features_to_shift):
    ax.hist(
        reference[feature],
        bins=20,
        alpha=0.6,
        label="Reference"
    )
    ax.hist(
        current[feature],
        bins=20,
        alpha=0.6,
        label="Current"
    )
    ax.set_title(feature)
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Frequency")
    ax.legend()

plt.tight_layout()
plot_path = "reports/drift_comparison.png"
plt.savefig(plot_path)
plt.close()

# 7. Log parameters, metrics, and artifacts in MLflow
with mlflow.start_run(run_name="Data_Drift_Detection"):
    mlflow.log_param("dataset", "Breast Cancer Wisconsin")
    mlflow.log_param("reference_rows", len(reference))
    mlflow.log_param("current_rows", len(current))
    mlflow.log_param("detection_method", "Kolmogorov-Smirnov test")
    mlflow.log_param("p_value_threshold", threshold)
    mlflow.log_param("drift_type", "Simulated data drift")

    drift_count = int(report["drift_detected"].sum())
    mlflow.log_metric("features_tested", len(report))
    mlflow.log_metric("features_with_drift", drift_count)

    for _, row in report.iterrows():
        safe_name = row["feature"].replace(" ", "_")
        mlflow.log_metric(
            f"ks_statistic_{safe_name}",
            float(row["ks_statistic"])
        )
        mlflow.log_metric(
            f"p_value_{safe_name}",
            float(row["p_value"])
        )

    mlflow.log_artifact(report_path)
    mlflow.log_artifact(plot_path)

print("\nData drift analysis completed successfully!")
print(f"Features tested: {len(report)}")
print(f"Features with detected drift: {drift_count}")
print("\nDrift report:")
print(
    report[report["drift_detected"]][
        ["feature", "ks_statistic", "p_value"]
    ].to_string(index=False)
)
print(f"\nReport saved to: {report_path}")
print(f"Plot saved to: {plot_path}")
