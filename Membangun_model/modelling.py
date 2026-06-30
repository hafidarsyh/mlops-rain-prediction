import os
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

warnings.filterwarnings("ignore")

# Konfigurasi DagsHub & MLflow
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "hafidarsyah")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "RainAUS-MLflow")

dagshub.init(
    repo_owner=DAGSHUB_USERNAME,
    repo_name=DAGSHUB_REPO,
    mlflow=True
)

# Load data
DATA_DIR = "weatherAUS_preprocessing"

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

print(f"Data dimuat:")
print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")


# Training dengan autlog
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

with mlflow.start_run(run_name="RF_Autolog_Baseline") as run:
    print(f"\nMemulai training... (Run ID: {run.info.run_id})")

    # Model dasar tanpa tuning
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluasi manual (akan ikut tercatat karena autolog aktif)
    y_pred      = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_pred_proba)

    print("\nHasil Evaluasi:")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")

print("\nTraining selesai! Cek MLflow Tracking di DagsHub.")
print(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
