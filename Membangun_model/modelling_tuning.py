import io
import os
import tempfile
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, cohen_kappa_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
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

mlflow.set_experiment("RainAUS-HyperparameterTuning")

# Load data
DATA_DIR = "weatherAUS_preprocessing"

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

print(f"Data dimuat:")
print(f"X_train : {X_train.shape}  |  Fitur: {X_train.shape[1]}")
print(f"X_test  : {X_test.shape}")


# Helper functions untuk membuat dan log artefak tambahan
def log_feature_importance(model, feature_names, top_n=20):
    """[ARTEFAK 1] Feature Importance Bar Chart."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))
    ax.barh(
        range(top_n), importances[idx][::-1],
        color=colors[::-1], edgecolor="white"
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]], fontsize=9)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.axvline(
        x=importances[idx].mean(), color="red",
        linestyle="--", alpha=0.7, label=f"Mean = {importances[idx].mean():.4f}"
    )
    ax.legend()
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(tmp.name, artifact_path="plots")
        os.unlink(tmp.name)
    plt.close(fig)
    print("[Artefak 1] Feature Importance chart di-log.")


def log_roc_curve(y_true, y_proba):
    """[ARTEFAK 2] ROC-AUC Curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC-AUC Curve — Rain in Australia", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(tmp.name, artifact_path="plots")
        os.unlink(tmp.name)
    plt.close(fig)
    print("[Artefak 2] ROC-AUC curve di-log.")


def log_confusion_matrix(y_true, y_pred):
    """[ARTEFAK 3] Confusion Matrix Heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    import seaborn as sns
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No Rain", "Rain"],
        yticklabels=["No Rain", "Rain"],
        annot_kws={"size": 13}
    )
    # Tambahkan persentase
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, f"({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=9, color="gray")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(tmp.name, artifact_path="plots")
        os.unlink(tmp.name)
    plt.close(fig)
    print("[Artefak 3] Confusion Matrix di-log.")


def log_classification_report(y_true, y_pred, run_id):
    """[ARTEFAK 4] Classification Report (file teks)."""
    report = classification_report(
        y_true, y_pred,
        target_names=["No Rain (0)", "Rain (1)"]
    )
    header = (
        "=" * 60 + "\n"
        f"  CLASSIFICATION REPORT\n"
        f"  Model : RandomForestClassifier (Tuned)\n"
        f"  Run ID: {run_id}\n"
        "=" * 60 + "\n\n"
    )
    full_report = header + report

    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(full_report)
        tmp_path = tmp.name

    mlflow.log_artifact(tmp_path, artifact_path="reports")
    os.unlink(tmp_path)
    print("[Artefak 4] Classification Report di-log.")
    return report


def log_precision_recall_curve(y_true, y_proba):
    """[ARTEFAK 5] Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#E91E63", lw=2,
            label=f"PR Curve (AP = {ap:.4f})")
    ax.axhline(y=baseline, color="gray", linestyle="--", lw=1.5,
               label=f"Baseline = {baseline:.4f}")
    ax.fill_between(recall, precision, baseline, alpha=0.1, color="#E91E63")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve — Rain in Australia",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(tmp.name, artifact_path="plots")
        os.unlink(tmp.name)
    plt.close(fig)
    print("[Artefak 5] Precision-Recall Curve di-log.")


# Hyperparameter space untuk Random Forest
param_distributions = {
    "n_estimators"      : [100, 200, 300, 500],
    "max_depth"         : [None, 10, 20, 30, 50],
    "min_samples_split" : [2, 5, 10, 15],
    "min_samples_leaf"  : [1, 2, 4, 8],
    "max_features"      : ["sqrt", "log2", 0.3, 0.5],
    "class_weight"      : ["balanced", None],
    "bootstrap"         : [True, False],
}

base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nMemulai Hyperparameter Tuning (RandomizedSearchCV)...")
print(f"n_iter    = 20")
print(f"cv        = 5-fold StratifiedKFold")
print(f"scoring   = f1")

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=20,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    random_state=42,
    verbose=1,
    return_train_score=True,
)

search.fit(X_train, y_train)

best_model  = search.best_estimator_
best_params = search.best_params_

print(f"\nTuning selesai!")
print(f"Best F1 (CV)     : {search.best_score_:.4f}")
print(f"Best Parameters  : {best_params}")


# Manual logging ke MLflow (melampaui autolog) untuk model yang sudah dituning
with mlflow.start_run(run_name="RF_ManualLog_Tuned_Advanced") as run:
    run_id = run.info.run_id
    print(f"\nLogging ke MLflow DagsHub... (Run ID: {run_id})")

    # ── PARAMETERS ───────────────────────────────────────────────────────────
    mlflow.log_param("model",              "RandomForestClassifier")
    mlflow.log_param("tuning_method",      "RandomizedSearchCV")
    mlflow.log_param("cv_folds",           5)
    mlflow.log_param("n_iter",             20)
    mlflow.log_param("scoring",            "f1")
    mlflow.log_param("random_state",       42)
    mlflow.log_param("test_size",          0.2)
    mlflow.log_param("dataset",            "Rain in Australia")
    mlflow.log_param("n_features",         X_train.shape[1])
    mlflow.log_param("n_train_samples",    len(X_train))
    mlflow.log_param("n_test_samples",     len(X_test))

    # Log best hyperparameters
    for k, v in best_params.items():
        mlflow.log_param(f"best_{k}", v)

    # METRICS (autolog-equivalent)
    y_pred       = best_model.predict(X_test)
    y_proba      = best_model.predict_proba(X_test)[:, 1]
    y_pred_train = best_model.predict(X_train)

    metrics = {
        # Test metrics
        "test_accuracy"        : accuracy_score(y_test, y_pred),
        "test_precision"       : precision_score(y_test, y_pred),
        "test_recall"          : recall_score(y_test, y_pred),
        "test_f1_score"        : f1_score(y_test, y_pred),
        "test_roc_auc"         : roc_auc_score(y_test, y_proba),
        "test_log_loss"        : log_loss(y_test, y_proba),
        "test_cohen_kappa"     : cohen_kappa_score(y_test, y_pred),
        # Train metrics
        "train_accuracy"       : accuracy_score(y_train, y_pred_train),
        "train_f1_score"       : f1_score(y_train, y_pred_train),
        # CV metrics dari tuning
        "cv_best_f1"           : search.best_score_,
        "cv_mean_train_score"  : search.cv_results_["mean_train_score"][search.best_index_],
        "cv_mean_test_score"   : search.cv_results_["mean_test_score"][search.best_index_],
        "cv_std_test_score"    : search.cv_results_["std_test_score"][search.best_index_],
    }

    mlflow.log_metrics(metrics)

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"{k:<30} : {v:.4f}")

    # LOG MODEL (autolog-equivalent
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="RainAUS_RandomForest"
    )
    print("\nModel artifact di-log.")

    # ARTEFAK TAMBAHAN (melampaui autolog)
    print("\nLogging artefak tambahan...")
    log_feature_importance(best_model, list(X_train.columns), top_n=20)
    log_roc_curve(y_test, y_proba)
    log_confusion_matrix(y_test, y_pred)
    report = log_classification_report(y_test, y_pred, run_id)
    log_precision_recall_curve(y_test, y_proba)

    # TAG
    mlflow.set_tags({
        "author"    : "Hafid",
        "dataset"   : "Rain in Australia",
        "framework" : "scikit-learn",
        "stage"     : "tuning",
        "level"     : "advanced"
    })

print("TRAINING & LOGGING SELESAI!")
print(f"\nRingkasan Hasil:")
print(f"Accuracy  : {metrics['test_accuracy']:.4f}")
print(f"Precision : {metrics['test_precision']:.4f}")
print(f"Recall    : {metrics['test_recall']:.4f}")
print(f"F1 Score  : {metrics['test_f1_score']:.4f}")
print(f"ROC-AUC   : {metrics['test_roc_auc']:.4f}")
print(f"\nCek MLflow di DagsHub:")
print(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
print(f"\nClassification Report:")
print(report)
