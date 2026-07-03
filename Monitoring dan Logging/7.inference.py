import os
import sys
import time
import json
import warnings
import importlib.util
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
from collections import deque

warnings.filterwarnings("ignore")


_exporter_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "3.prometheus_exporter.py"
)
_spec = importlib.util.spec_from_file_location("prometheus_exporter_module", _exporter_path)
_prometheus_exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prometheus_exporter)

start_exporter        = _prometheus_exporter.start_exporter
record_prediction     = _prometheus_exporter.record_prediction
record_error          = _prometheus_exporter.record_error
update_model_metrics  = _prometheus_exporter.update_model_metrics
update_drift          = _prometheus_exporter.update_drift
ACTIVE_REQUESTS       = _prometheus_exporter.ACTIVE_REQUESTS
simulate_drift_score  = _prometheus_exporter.simulate_drift_score


# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MODEL_URI           = os.getenv("MODEL_URI", "models:/RainAUS_RF_CI/latest")
EXPORTER_PORT        = int(os.getenv("EXPORTER_PORT", "8001"))

WINDOW_SIZE   = 100
recent_preds  = deque(maxlen=WINDOW_SIZE)
recent_inputs = deque(maxlen=WINDOW_SIZE)


# Load model
model = None

def load_model():
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
        print(f"Model dimuat dari: {MODEL_URI}")
    except Exception as e:
        print(f"Gagal load dari {MODEL_URI}: {e}")
        print("Mencari model terbaru di mlruns...")
        try:
            client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            exps = client.search_experiments()
            for exp in exps:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1
                )
                if runs:
                    run_id   = runs[0].info.run_id
                    fallback = f"runs:/{run_id}/model"
                    model    = mlflow.sklearn.load_model(fallback)
                    print(f"Model dimuat dari fallback: {fallback}")
                    break
        except Exception as e2:
            print(f"Gagal load model: {e2}")
            print("Pastikan mlruns/ tersedia dan berisi model terlatih.")
            sys.exit(1)


# Pydantic schemas
class PredictInput(BaseModel):
    MinTemp          : float = Field(...)
    MaxTemp          : float = Field(...)
    Rainfall         : float = Field(default=0.0)
    Evaporation      : float = Field(default=5.0)
    Sunshine         : float = Field(default=8.0)
    WindGustSpeed    : float = Field(default=40.0)
    WindSpeed9am     : float = Field(default=15.0)
    WindSpeed3pm     : float = Field(default=20.0)
    Humidity9am      : float = Field(default=70.0)
    Humidity3pm      : float = Field(default=50.0)
    Pressure9am      : float = Field(default=1015.0)
    Pressure3pm      : float = Field(default=1012.0)
    Cloud9am         : float = Field(default=4.0)
    Cloud3pm         : float = Field(default=4.0)
    Temp9am          : float = Field(default=18.0)
    Temp3pm          : float = Field(default=25.0)
    RainToday        : int   = Field(default=0, ge=0, le=1)
    Year             : int   = Field(default=2024)
    Month            : int   = Field(default=6, ge=1, le=12)
    Day              : int   = Field(default=15, ge=1, le=31)
    Location         : int   = Field(default=0)
    WindGustDir      : int   = Field(default=0)
    WindDir9am       : int   = Field(default=0)
    WindDir3pm       : int   = Field(default=0)
    actual_label     : Optional[int] = Field(default=None, ge=0, le=1)


class BatchPredictInput(BaseModel):
    data: List[PredictInput]


class PredictResponse(BaseModel):
    prediction       : int
    probability_rain : float
    label            : str
    latency_ms       : float
    model_version    : str


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Memulai inference server...")
    load_model()
    start_exporter(port=EXPORTER_PORT)
    yield
    print("Inference server dihentikan.")


app = FastAPI(
    title       = "Rain in Australia - ML Inference API",
    description = "Serving model prediksi curah hujan Australia dengan monitoring Prometheus",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# Helper
def _update_rolling_metrics(y_pred, y_proba, actual=None):
    recent_inputs.append(y_proba)

    if actual is not None:
        recent_preds.append((y_pred, actual, y_proba))

    if len(recent_preds) >= 10:
        preds   = [r[0] for r in recent_preds]
        actuals = [r[1] for r in recent_preds]
        probas  = [r[2] for r in recent_preds]

        from sklearn.metrics import (
            accuracy_score, precision_score,
            recall_score, f1_score, roc_auc_score
        )
        try:
            update_model_metrics(
                accuracy  = accuracy_score(actuals, preds),
                precision = precision_score(actuals, preds, zero_division=0),
                recall    = recall_score(actuals, preds, zero_division=0),
                f1        = f1_score(actuals, preds, zero_division=0),
                roc_auc   = roc_auc_score(actuals, probas)
                            if len(set(actuals)) > 1 else 0.5,
            )
        except Exception:
            pass

    drift = simulate_drift_score(
        recent_means  = list(recent_inputs),
        baseline_mean = 0.5,
        threshold     = 0.3
    )
    update_drift(drift)


# Endpoints
@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status"       : "healthy",
        "model_loaded" : model is not None,
        "window_size"  : len(recent_preds),
    }


@app.get("/metrics-info", tags=["System"])
async def metrics_info():
    return {
        "requests_in_window" : len(recent_preds),
        "window_max_size"    : WINDOW_SIZE,
        "prometheus_metrics" : f"http://localhost:{EXPORTER_PORT}/metrics",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(payload: PredictInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat.")

    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        input_dict = payload.model_dump(exclude={"actual_label"})
        df = pd.DataFrame([input_dict])

        if hasattr(model, "feature_names_in_"):
            try:
                df = df[model.feature_names_in_]
            except KeyError as e:
                raise ValueError(f"Kolom input tidak cocok dengan model training. Detail: {e}")

        y_pred  = int(model.predict(df)[0])
        y_proba = float(model.predict_proba(df)[0][1])

        latency = time.time() - start_time
        record_prediction(y_pred, y_proba, latency)
        _update_rolling_metrics(y_pred, y_proba, actual=payload.actual_label)

        return PredictResponse(
            prediction       = y_pred,
            probability_rain = round(y_proba, 4),
            label            = "Hujan" if y_pred == 1 else "Tidak Hujan",
            latency_ms       = round(latency * 1000, 2),
            model_version    = MODEL_URI,
        )

    except Exception as e:
        record_error(error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/predict/batch", tags=["Inference"])
async def predict_batch(payload: BatchPredictInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat.")
    if len(payload.data) == 0:
        raise HTTPException(status_code=400, detail="Data batch kosong.")

    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        results = []
        for item in payload.data:
            input_dict = item.model_dump(exclude={"actual_label"})
            df      = pd.DataFrame([input_dict])

            if hasattr(model, "feature_names_in_"):
                try:
                    df = df[model.feature_names_in_]
                except KeyError as e:
                    raise ValueError(f"Kolom input tidak cocok dengan model training. Detail: {e}")

            y_pred  = int(model.predict(df)[0])
            y_proba = float(model.predict_proba(df)[0][1])

            record_prediction(y_pred, y_proba, 0)
            _update_rolling_metrics(y_pred, y_proba, actual=item.actual_label)

            results.append({
                "prediction"       : y_pred,
                "probability_rain" : round(y_proba, 4),
                "label"            : "Hujan" if y_pred == 1 else "Tidak Hujan",
            })

        total_latency = time.time() - start_time
        return {
            "count"       : len(results),
            "latency_ms"  : round(total_latency * 1000, 2),
            "predictions" : results,
        }

    except Exception as e:
        record_error(error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        ACTIVE_REQUESTS.dec()

# Main
if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("INFERENCE_PORT", "5000"))
    print(f"  Inference server: http://localhost:{PORT}")
    print(f"  API Docs       : http://localhost:{PORT}/docs")
    print(f"  Prometheus     : http://localhost:{EXPORTER_PORT}/metrics")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
