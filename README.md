# End-to-End MLOps Pipeline: Rain in Australia Prediction 🌧️⚡

[![MLflow](https://img.shields.io/badge/MLflow-v2.19.0-blue?style=flat-square&logo=mlflow)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue?style=flat-square&logo=docker)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-orange?style=flat-square&logo=githubactions)](https://github.com/features/actions)
[![DagsHub](https://img.shields.io/badge/Remote_Tracking-DagsHub-green?style=flat-square)](https://dagshub.com/)
[![Prometheus](https://img.shields.io/badge/Monitoring-Prometheus-red?style=flat-square&logo=prometheus)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Dashboard-Grafana-orange?style=flat-square&logo=grafana)](https://grafana.com/)

Repositori ini berisi implementasi arsitektur **Machine Learning Operations (MLOps) *end-to-end* yang siap produksi** untuk memprediksi probabilitas terjadinya hujan keesokan harinya di Australia (`RainTomorrow`).

Proyek ini merangkum seluruh siklus hidup pengembangan sistem *Machine Learning* modern: mulai dari otomasi *data preprocessing*, pelacakan eksperimen intensif (*experiment tracking*), manajemen model (*model registry*), integrasi *Continuous Integration/Continuous Deployment* (CI/CD), pengemasan kontainer (*containerization*), hingga penyajian layanan (*model serving*) dan pemantauan sistem secara waktu nyata (*real-time monitoring & alerting*).

---

## 🏗️ Arsitektur & Fitur Utama

```text
[Raw Dataset] ──> [Automated Preprocessing] ──> [Model Training & Tuning]
                                                           │
                                                           ▼
[Docker Hub] <── [GitHub Actions CI/CD] <── [MLflow & DagsHub Tracking]
      │
      ▼
[FastAPI Serving] ──> [Prometheus Exporter] ──> [Grafana Dashboard & Alerts]
```

1. **Automated Data Pipeline (`preprocessing/`)**
   - Membersihkan data mentah (*handling missing values & outliers*) dan melakukan rekayasa fitur (*feature engineering*).
   - Skrip modular `automate_Hafid.py` untuk mengotomatisasi transformasi data mentah menjadi dataset siap latih secara konsisten dan dapat direproduksi.

2. **Experiment Tracking & Versioning (`Membangun_model/`)**
   - Terintegrasi dengan **MLflow** untuk mencatat parameter, metrik evaluasi (*Accuracy*, *F1-Score*, *Precision*, *Recall*, *ROC-AUC*, *Log Loss*), dan artefak grafis (kurva ROC, *Confusion Matrix*, *Feature Importance*).
   - Mendukung pelacakan kolaboratif berbasis *cloud* dengan menghubungkan MLflow lokal ke repositori **DagsHub**.

3. **CI/CD Pipeline & Containerization (`Workflow-CI/` & `.github/`)**
   - Standardisasi lingkungan eksekusi menggunakan **MLflow Project** dan konfigurasi terisolasi `conda.yaml`.
   - Workflow **GitHub Actions** yang secara otomatis memicu pelatihan ulang (*re-training*) saat terjadi perubahan pada dataset atau pipeline pemodelan.
   - Otomatisasi pengemasan model menjadi Docker Image siap produksi menggunakan fungsi `mlflow build-docker` dan mengunggahnya langsung ke **Docker Hub**.

4. **Real-Time Model Serving (`Monitoring dan Logging/`)**
   - Layanan inferensi cepat berkinerja tinggi menggunakan framework **FastAPI** yang dikemas dalam kontainer Docker.
   - Menyediakan REST API dengan dokumentasi interaktif (Swagger UI) untuk integrasi dengan aplikasi klien.

5. **Active Observability & Alerting Stack (`Monitoring dan Logging/`)**
   - Pengumpulan metrik kustom berbasis `prometheus_exporter.py` untuk mengukur latensi prediksi, *request rate*, pemanfaatan sumber daya (*CPU/Memory*), serta estimasi *real-time model accuracy* dan **Data Drift**.
   - Visualisasi mendalam menggunakan dasbor interaktif **Grafana**.
   - Sistem **Alerting** otomatis yang akan memicu notifikasi bahaya apabila terjadi degradasi performa model (*accuracy drop*) atau penyimpangan distribusi fitur (*data drift*) yang melampaui ambang batas kritis.

---

## 📂 Struktur Repositori

```text
mlops-rain-prediction/
├── .github/workflows/
│   └── ci.yml                          # Workflow GitHub Actions untuk otomatisasi CI/CD & Docker build
├── dataset/
│   └── weatherAUS.csv                  # Dataset mentah (Raw Data) Rain in Australia
├── preprocessing/
│   ├── Eksperimen_Hafid.ipynb          # Notebook riset, EDA, dan eksperimen awal
│   ├── automate_Hafid.py               # Skrip otomasi pemrosesan dataset menjadi siap latih
│   └── weatherAUS_preprocessing/       # Output direktori data pemrosesan (X_train, y_train, dst.)
├── Membangun_model/
│   ├── modelling.py                    # Skrip pelatihan model dasar dengan logging MLflow
│   ├── modelling_tuning.py             # Skrip lanjutan: hyperparameter tuning + pelacakan ke DagsHub
│   ├── requirements.txt                # Dependensi lingkungan pemodelan
│   └── DagsHub.txt                     # Referensi tautan repositori remote tracking DagsHub
├── Workflow-CI/
│   ├── ci.yml                          # Salinan konfigurasi alur kerja CI/CD
│   ├── mlflow.db                       # Database lokal MLflow
│   └── MLProject/                      # Bundel MLflow Project untuk eksekusi pipeline yang reprodusibel
│       ├── MLProject                   # File konfigurasi entry point MLflow Project
│       ├── conda.yaml                  # Spesifikasi lingkungan (dependencies isolation)
│       ├── modelling.py                # Eksekutor utama pelatihan di dalam pipeline CI
│       └── docker_hub_link.txt         # Referensi URL kontainer imej produksi di Docker Hub
├── Monitoring dan Logging/
│   ├── 2.prometheus.yml                # Konfigurasi target scraping Prometheus
│   ├── 3.prometheus_exporter.py        # Custom exporter metrik performa model, latensi, dan data drift
│   ├── 7.inference.py                  # API Gateway inferensi menggunakan FastAPI
│   ├── Dockerfile.inference            # Instruksi pembuatan kontainer layanan inferensi
│   ├── docker-compose.yml              # Orkestrasi penuh: FastAPI + Exporter + Prometheus + Grafana
│   ├── generate-traffic-test-alert.py  # Skrip simulasi trafik untuk pengujian load & data drift alerting
│   ├── grafana_dashboard.json          # Konfigurasi ekspor dasbor Grafana
│   └── grafana_alerts.json             # Konfigurasi aturan alerting Grafana
├── requirements.txt                    # Dependensi global proyek
└── LICENSE                             # Lisensi MIT
```
*(Catatan: Folder seperti `screenshoot_*`, `4.bukti_*`, `5.bukti_*`, dan `6.bukti_*` merupakan arsip dokumentasi visual performa sistem saat dieksekusi).*

---

## 🚀 Panduan Instalasi & Quickstart

### 1. Prasyarat Sistem
Pastikan lingkungan Anda telah terpasang:
- **Python >= 3.12**
- **Docker** & **Docker Compose**
- **Git**

### 2. Kloning Repositori & Instalasi Dependensi
```bash
git clone https://github.com/hafidarsyh/mlops-rain-prediction.git
cd mlops-rain-prediction
pip install -r requirements.txt
```

### 3. Menjalankan Otomasi Preprocessing
Untuk mengolah data mentah `weatherAUS.csv` menjadi dataset matang:
```bash
python preprocessing/automate_Hafid.py
```

### 4. Pelatihan Model & Experiment Tracking
Anda dapat menjalankan pelatihan model di lokal sekaligus mengirimkan log metrik dan artefak ke server remote **DagsHub**:
```bash
# Instalasi & login CLI DagsHub (jika menggunakan remote tracking)
pip install dagshub
dagshub login

# Eksekusi pelatihan dengan hyperparameter tuning
python Membangun_model/modelling_tuning.py
```
Akses MLflow UI secara lokal dengan menjalankan `mlflow ui` di terminal dan buka [http://localhost:5000](http://localhost:5000).

### 5. Menjalankan Production Serving & Observability Stack
Seluruh ekosistem produksi (FastAPI API Server, Prometheus Exporter, Prometheus Server, dan Grafana Dashboard) dapat diangkat dalam satu perintah menggunakan Docker Compose:
```bash
cd "Monitoring dan Logging"
docker-compose up --build -d
```

Daftar layanan yang dapat diakses:
- **FastAPI Inference Server:** [http://localhost:8000](http://localhost:8000)
- **FastAPI Interactive API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Prometheus UI:** [http://localhost:9090](http://localhost:9090)
- **Grafana Dashboard:** [http://localhost:3000](http://localhost:3000) *(Login default: admin / admin)*

---

## 📈 Pemantauan Metrik & Simulasi Alerting

Sistem observabilitas ini secara kontinyu memantau performa teknis dan analitis model, meliputi:
- **System Metrics:** Utilization CPU, Memory Usage, HTTP Request Counters (2xx, 4xx, 5xx), dan Latency Profile (P95/P99).
- **ML Performance Metrics:** Real-time Model Accuracy & F1-Score Trend.
- **Data Quality & Drift:** Pengukuran pergeseran distribusi *input features* (*Data Drift Score*) dan tingkat nilai kosong (*Missing Values Rate*).

### Menguji Sistem Alerting
Untuk memverifikasi fungsi pemantauan, Anda dapat menjalankan skrip simulasi lalu lintas data yang akan menembakkan *request* normal serta *anomaly request* (untuk memicu kondisi *Data Drift* dan penurunan akurasi):

```bash
cd "Monitoring dan Logging"
python generate-traffic-test-alert.py
```
Setelah skrip berjalan beberapa saat, dasbor Grafana akan menampilkan perubahan lonjakan trafik, dan sistem Alerting akan otomatis berubah berstatus **ALERTIING** / **FIRING** saat ambang batas kritis terlampaui.

---

## 🛠️ Eksekusi MLflow Project CLI

Untuk menjalankan pipeline pelatihan dalam mode *reproducible environment* lokal menggunakan MLflow Project:
```bash
mlflow run Workflow-CI/MLProject --no-conda
```

---

## 👤 Pengembang

**Hafid Ardiansyah**
- *Web Developer & MLOps Practitioner*
- GitHub: [@hafidarsyh](https://github.com/hafidarsyh)

---

## 📄 Lisensi

Proyek ini berada di bawah lisensi **MIT License**. Lihat berkas [LICENSE](LICENSE) untuk detail lebih lanjut.
