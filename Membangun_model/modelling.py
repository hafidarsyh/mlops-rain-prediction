import argparse
import os
import sys
import tempfile
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, cohen_kappa_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RandomForestClassifier on Rain in Australia dataset"
    )
    parser.add_argument("--n_estimators",      type=int,   default=200)
    parser.add_argument("--max_depth",         type=int,   default=20)
    parser.add_argument("--min_samples_split", type=int,   default=5)
    parser.add_argument("--min_samples_leaf",  type=int,   default=2)
    parser.add_argument("--max_features",      type=str,   default="sqrt")
    parser.add_argument("--test_size",         type=float, default=0.2)
    parser.add_argument("--random_state",      type=int,   default=42)
    return parser.parse_args()


def log_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))
    ax.barh(range(top_n), importances[idx][::-1],
            color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]], fontsize=9)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.axvline(x=importances[idx].mean(), color="red", linestyle="--", alpha=0.7,
               label=f"Mean = {importances[idx].mean():.4f}")
    ax.legend()
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(tmp.name, artifact_path="plots")
        os.unlink(tmp.name)
    plt.close(fig)


def log_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    ax.set(xlim=[0, 1], ylim=[0, 1.05],
           xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC-AUC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(tmp.name, artifact_path="plots")
        os.unlink(tmp.name)
    plt.close(fig)


def log_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Rain", "Rain"],
                yticklabels=["No Rain", "Rain"])
    ax.set_title("Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(tmp.name, artifact_path="plots")
        os.unlink(tmp.name)
    plt.close(fig)


def log_classification_report(y_true, y_pred, run_id):
    report = classification_report(
        y_true, y_pred,
        target_names=["No Rain (0)", "Rain (1)"]
    )
    content = (
        "=" * 60 + "\n"
        f"CLASSIFICATION REPORT\n"
        f"Run ID : {run_id}\n"
        "=" * 60 + "\n\n"
        + report
    )
    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    mlflow.log_artifact(tmp_path, artifact_path="reports")
    os.unlink(tmp_path)


def log_precision_recall_curve(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#E91E63", lw=2, label=f"AP = {ap:.4f}")
    ax.axhline(y=y_true.mean(), color="gray", linestyle="--", lw=1.5,
               label=f"Baseline = {y_true.mean():.4f}")
    ax.set(xlim=[0, 1], ylim=[0, 1.05],
           xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curve")
    ax.legend()
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(tmp.name, artifact_path="plots")
        os.unlink(tmp.name)
    plt.close(fig)


def main():
    args = parse_args()

    dagshub_user = os.getenv("DAGSHUB_USERNAME", "hafidarsyah")
    dagshub_repo = os.getenv("DAGSHUB_REPO", "RainAUS-MLflow")

    if dagshub_user and dagshub_repo:
        import dagshub
        dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)
        print(f"MLflow - DagsHub ({dagshub_user}/{dagshub_repo})")
    else:
        mlflow.set_tracking_uri("mlruns")
        print("MLflow - local (mlruns/)")

    mlflow.set_experiment("RainAUS-CI-Training")

    DATA_DIR = os.path.join(os.path.dirname(__file__), "weatherAUS_preprocessing")
    try:
        X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
        X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
        y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()
    except FileNotFoundError as e:
        print(f"Data tidak ditemukan: {e}")
        print("Pastikan folder weatherAUS_preprocessing/ ada di dalam Folder/")
        sys.exit(1)

    print(f"Data: X_train={X_train.shape}, X_test={X_test.shape}")

    # ── Training + Manual Logging ─────────────────────────────────────────
    with mlflow.start_run(run_name="RF_CI_ManualLog") as run:
        run_id = run.info.run_id
        print(f"\nRun ID: {run_id}")

        # Parameters
        params = {
            "model"             : "RandomForestClassifier",
            "n_estimators"      : args.n_estimators,
            "max_depth"         : args.max_depth,
            "min_samples_split" : args.min_samples_split,
            "min_samples_leaf"  : args.min_samples_leaf,
            "max_features"      : args.max_features,
            "test_size"         : args.test_size,
            "random_state"      : args.random_state,
            "n_features"        : X_train.shape[1],
            "n_train_samples"   : len(X_train),
            "n_test_samples"    : len(X_test),
        }
        mlflow.log_params(params)

        # Training
        model = RandomForestClassifier(
            n_estimators      = args.n_estimators,
            max_depth         = args.max_depth if args.max_depth > 0 else None,
            min_samples_split = args.min_samples_split,
            min_samples_leaf  = args.min_samples_leaf,
            max_features      = args.max_features,
            random_state      = args.random_state,
            class_weight      = "balanced",
            n_jobs            = -1,
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred        = model.predict(X_test)
        y_proba       = model.predict_proba(X_test)[:, 1]
        y_pred_train  = model.predict(X_train)

        # Metrics (manual logging)
        metrics = {
            "test_accuracy"    : accuracy_score(y_test, y_pred),
            "test_precision"   : precision_score(y_test, y_pred),
            "test_recall"      : recall_score(y_test, y_pred),
            "test_f1_score"    : f1_score(y_test, y_pred),
            "test_roc_auc"     : roc_auc_score(y_test, y_proba),
            "test_log_loss"    : log_loss(y_test, y_proba),
            "test_cohen_kappa" : cohen_kappa_score(y_test, y_pred),
            "train_accuracy"   : accuracy_score(y_train, y_pred_train),
            "train_f1_score"   : f1_score(y_train, y_pred_train),
        }
        mlflow.log_metrics(metrics)

        # Model artifact
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="RainAUS_RF_CI"
        )

        # Extra artifacts
        log_feature_importance(model, list(X_train.columns))
        log_roc_curve(y_test, y_proba)
        log_confusion_matrix(y_test, y_pred)
        log_classification_report(y_test, y_pred, run_id)
        log_precision_recall_curve(y_test, y_proba)

        # Tags
        mlflow.set_tags({
            "author"    : "Hafid",
            "trigger"   : os.getenv("GITHUB_EVENT_NAME", "local"),
            "commit_sha": os.getenv("GITHUB_SHA", "local"),
            "stage"     : "CI-production",
        })

        # Simpan run_id ke file agar bisa dibaca oleh workflow CI
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        print(f"\nrun_id.txt disimpan: {run_id}")

    print("\n" + "=" * 55)
    print("  TRAINING SELESAI")
    print("=" * 55)
    for k, v in metrics.items():
        print(f"  {k:<25} : {v:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
