# Predictive Resource Scaler

> An ML-powered server autoscaling simulation that predicts CPU spikes before they happen, demonstrating measurable cost savings over reactive scaling strategies.

---

## The Problem

Traditional cloud autoscaling is **reactive** — it waits until CPU is already high, then boots a new server instance. Since booting takes ~3 minutes, users experience degraded performance during that window. Every minute of CPU above 80% is an SLA breach — costing money and user trust.

This project builds a **proactive scaler** using Machine Learning:
1. Trains a model on historical server traffic to **predict CPU spikes 5 minutes ahead**
2. Fires scale-up events early so instances are **ready before the spike hits**
3. Runs a **minute-by-minute cost simulation** comparing both strategies with a real pricing model

---

## Results

| Metric | Value |
|---|---|
| Random Forest spike prediction precision | **~87%** |
| Random Forest MAE | **8.2% CPU** |
| Linear Regression MAE | 8.8% CPU |
| RF R² score | **0.70** |
| Decision threshold tuned for precision | 0.78 |

> Run `python main.py` to reproduce all results locally.

---

## Architecture

```
generate_data.py   →  90-day synthetic server logs (129,600 rows, 1 per minute)
data_prep.py       →  21 features: lag, rolling averages, time features
train_models.py    →  Linear Regression vs Random Forest — comparison + evaluation
simulate.py        →  Minute-by-minute reactive vs proactive simulation with cost model
visualize.py       →  5 Matplotlib charts
main.py            →  Full pipeline in one command
```

---

## Charts Generated

| Chart | What It Shows |
|---|---|
| `01_raw_cpu_pattern.png` | Daily CPU cycles and hourly spike zones |
| `02_actual_vs_predicted.png` | Model accuracy — predicted vs actual CPU |
| `03_reactive_vs_proactive.png` | Side-by-side behaviour during spike events |
| `04_cost_comparison.png` | SLA breach cost + instance cost breakdown |
| `05_feature_importance.png` | Which features the Random Forest relied on most |

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, time-series feature engineering |
| `numpy` | Numerical operations, synthetic data generation |
| `scikit-learn` | LinearRegression, RandomForestRegressor, RandomForestClassifier, metrics |
| `matplotlib` | All 5 visualizations |

No cloud infrastructure required — runs entirely locally.

---

## Setup & Run

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/predictive-resource-scaler.git
cd predictive-resource-scaler

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run full pipeline
python main.py
```

Individual steps:

```bash
python generate_data.py    # Step 1: Creates data/server_logs.csv
python data_prep.py        # Step 2: Feature engineering
python train_models.py     # Step 3: Train + evaluate models
python simulate.py         # Step 4: Run scaling simulation
python visualize.py        # Step 5: Generate all charts
```

---

## Key Technical Decisions

### Why Random Forest over Linear Regression?
CPU spikes are caused by **non-linear interactions** between features — "Friday evening AND high 5-minute rolling average AND high lag features" together signal a spike. Linear Regression treats each feature independently and misses these combinations. Random Forest decision trees naturally capture feature interactions through their splitting mechanism. Feature importance analysis confirmed `cpu_usage` (75.7%) and `cpu_rolling_mean_30` (9.7%) dominate — the model learned to track CPU momentum, not schedules.

### Why NOT use sklearn's random train/test split?
For time series, random splitting causes **data leakage** — the model trains on January 15 data and tests on January 10, effectively seeing the future. This inflates all metrics. This project uses a strict **chronological split**: first 80% of timestamps for training, last 20% for testing. The model never sees any data from after its training cutoff.

### Why a custom decision threshold of 0.78?
sklearn's default threshold of 0.50 fires too many false alarms in production — an autoscaling system that cries wolf gets turned off by operators. Using `predict_proba()[:,1]` with a tuned threshold of 0.78 means the system only warns when it is 78%+ confident, achieving ~87% precision. When the system fires a warning, it is almost always right.

### Why synthetic data?
Real production server logs are confidential. The synthetic generator uses **sine waves** for daily/weekly traffic cycles, **Gaussian noise** for minute-to-minute variation, and **random burst injection** for spike events — producing statistically realistic data without privacy concerns.

---

## Feature Engineering

21 features engineered from raw CPU readings:

| Feature Type | Features Created | Why |
|---|---|---|
| Lag features | `cpu_lag_1`, `cpu_lag_5`, `cpu_lag_10`, `cpu_lag_30` | Captures CPU trend direction |
| Rolling averages | `cpu_rolling_mean_5`, `cpu_rolling_mean_15`, `cpu_rolling_mean_30` | Smooths noise, reveals true trend |
| Rolling std | `cpu_rolling_std_5`, `cpu_rolling_std_15` | Volatility = instability signal |
| Time features | `hour`, `day_of_week`, `is_weekend`, `month` | Periodic traffic patterns |
| Raw metrics | `cpu_usage`, `request_count`, `memory_usage` | Direct server state |

---

## Simulation Cost Model

| Parameter | Value | Basis |
|---|---|---|
| Instance running cost | $0.02/min | Approximate AWS t3.medium |
| SLA breach penalty | $0.50/min | Lost customer value estimate |
| Instance boot time | 3 minutes | Typical EC2 cold start |
| Scale-down cooldown | 15 minutes | Prevents rapid flapping |
| Proactive trigger threshold | 35% spike probability | Balances early warning vs false alarms |

---

## Productionisation Path

1. Replace CSV with **Prometheus / CloudWatch real-time metrics stream**
2. Containerise prediction service with **Docker**
3. Expose via **FastAPI REST endpoint**
4. Integrate with **Kubernetes HPA** custom metrics
5. Schedule weekly **model retraining** via Airflow

---

## Author

**Dev Sharma** — B.Tech Mechanical Engineering, IIT Guwahati