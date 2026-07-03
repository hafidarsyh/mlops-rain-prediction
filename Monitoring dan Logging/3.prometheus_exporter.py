import os
import time
import threading
import psutil
import numpy as np
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    start_http_server, REGISTRY, CollectorRegistry
)

REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total jumlah request prediksi yang masuk",
    labelnames=["status"]
)

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Distribusi waktu prediksi (detik)",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

PREDICTIONS_RAIN = Counter(
    "predictions_rain_total",
    "Total prediksi RainTomorrow = 1 (akan hujan)"
)

PREDICTIONS_NO_RAIN = Counter(
    "predictions_no_rain_total",
    "Total prediksi RainTomorrow = 0 (tidak hujan)"
)

CONFIDENCE_AVG = Gauge(
    "prediction_confidence_avg",
    "Rata-rata skor kepercayaan (probabilitas) prediksi terakhir"
)

CONFIDENCE_HISTOGRAM = Histogram(
    "prediction_confidence",
    "Distribusi skor kepercayaan prediksi",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

ERROR_REQUESTS = Counter(
    "error_requests_total",
    "Total request yang menghasilkan error",
    labelnames=["error_type"]
)

ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Jumlah request yang sedang diproses saat ini"
)

MODEL_ACCURACY = Gauge(
    "model_accuracy_score",
    "Akurasi model (rolling window pada request terakhir)"
)

MODEL_PRECISION = Gauge(
    "model_precision_score",
    "Precision model (rolling window)"
)

MODEL_RECALL = Gauge(
    "model_recall_score",
    "Recall model (rolling window)"
)

MODEL_F1 = Gauge(
    "model_f1_score",
    "F1 Score model (rolling window)"
)

MODEL_ROC_AUC = Gauge(
    "model_roc_auc_score",
    "ROC-AUC Score model (rolling window)"
)

DATA_DRIFT_SCORE = Gauge(
    "data_drift_score",
    "Skor drift data input dibandingkan data training (0=no drift, 1=high drift)"
)

CPU_USAGE = Gauge(
    "system_cpu_usage_percent",
    "Persentase penggunaan CPU"
)

MEMORY_USAGE = Gauge(
    "system_memory_usage_bytes",
    "Penggunaan memori dalam bytes"
)

MEMORY_USAGE_PERCENT = Gauge(
    "system_memory_usage_percent",
    "Persentase penggunaan memori"
)

DISK_USAGE_PERCENT = Gauge(
    "system_disk_usage_percent",
    "Persentase penggunaan disk"
)


# Helper
def update_system_metrics():
    """Update metrik sistem secara periodik."""
    CPU_USAGE.set(psutil.cpu_percent(interval=1))

    mem = psutil.virtual_memory()
    MEMORY_USAGE.set(mem.used)
    MEMORY_USAGE_PERCENT.set(mem.percent)

    disk = psutil.disk_usage("/")
    DISK_USAGE_PERCENT.set(disk.percent)


def simulate_drift_score(recent_means: list, baseline_mean: float = 0.0,
                         threshold: float = 0.5) -> float:
    """
    Hitung drift score sederhana berdasarkan pergeseran mean fitur input.
    Nilai mendekati 1.0 = drift tinggi.
    """
    if not recent_means:
        return 0.0
    current_mean = np.mean(recent_means)
    drift = abs(current_mean - baseline_mean) / (threshold + 1e-8)
    return min(float(drift), 1.0)


def record_prediction(y_pred: int, y_proba: float, latency: float):
    """
    Catat satu prediksi ke semua metrik yang relevan.

    Parameters
    ----------
    y_pred   : int    — kelas prediksi (0 atau 1)
    y_proba  : float  — probabilitas kelas positif
    latency  : float  — waktu inferensi dalam detik
    """
    REQUESTS_TOTAL.labels(status="success").inc()
    INFERENCE_LATENCY.observe(latency)
    CONFIDENCE_HISTOGRAM.observe(y_proba)
    CONFIDENCE_AVG.set(y_proba)

    if y_pred == 1:
        PREDICTIONS_RAIN.inc()
    else:
        PREDICTIONS_NO_RAIN.inc()


def record_error(error_type: str = "unknown"):
    """Catat request yang error."""
    REQUESTS_TOTAL.labels(status="error").inc()
    ERROR_REQUESTS.labels(error_type=error_type).inc()


def update_model_metrics(accuracy: float, precision: float,
                         recall: float, f1: float, roc_auc: float):
    """Update metrik performa model."""
    MODEL_ACCURACY.set(accuracy)
    MODEL_PRECISION.set(precision)
    MODEL_RECALL.set(recall)
    MODEL_F1.set(f1)
    MODEL_ROC_AUC.set(roc_auc)


def update_drift(score: float):
    """Update skor data drift."""
    DATA_DRIFT_SCORE.set(score)


# Background Thread
def _system_metrics_loop(interval: int = 10):
    while True:
        try:
            update_system_metrics()
        except Exception as e:
            print(f"[EXPORTER] Error update system metrics: {e}")
        time.sleep(interval)


def start_exporter(port: int = 8001):
    """
    Mulai HTTP server Prometheus exporter.
    Metrics tersedia di http://localhost:{port}/metrics
    """
    start_http_server(port)
    print(f"Prometheus exporter berjalan di http://localhost:{port}/metrics")

    t = threading.Thread(target=_system_metrics_loop, daemon=True)
    t.start()

    update_model_metrics(
        accuracy  = float(os.getenv("INIT_ACCURACY",  "0.8500")),
        precision = float(os.getenv("INIT_PRECISION", "0.7200")),
        recall    = float(os.getenv("INIT_RECALL",    "0.6800")),
        f1        = float(os.getenv("INIT_F1",        "0.6990")),
        roc_auc   = float(os.getenv("INIT_ROC_AUC",  "0.8800")),
    )
    DATA_DRIFT_SCORE.set(0.0)
    print("Nilai awal model metrics berhasil diset.")


# Main
if __name__ == "__main__":
    PORT = int(os.getenv("EXPORTER_PORT", "8001"))
    start_exporter(port=PORT)

    print(f"\n  Metrik tersedia di: http://localhost:{PORT}/metrics")
    print("   Tekan Ctrl+C untuk berhenti.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Exporter dihentikan.")
