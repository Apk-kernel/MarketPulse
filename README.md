# MarketPulse: Hierarchical Probabilistic Demand Forecasting Engine

MarketPulse is a production-oriented, research-grade e-commerce demand forecasting and benchmarking engine inspired by demand-planning architectures at Walmart, Amazon, and Flipkart.

The system uses `sktime` as its primary time-series forecasting framework, coupled with rigorous statistical validation, hierarchical reconciliation, and probabilistic interval forecasting.

---

## 1. System Architecture

```
MarketPulse/
├── configs/
│   └── config.yaml               # Central YAML configuration
├── data/
│   ├── raw/                      # Original M5 CSV files
│   ├── processed/                # Memory-optimized Parquet files and audit reports
│   └── external/                 # External promotional/holiday calendars
├── experiments/                  # Experiment tracking, artifacts, and benchmarking reports
├── notebooks/                    # Research EDA and modeling notebooks
├── src/
│   ├── data/                     # Ingestion, validation, preprocessing, and hierarchy
│   ├── features/                 # Temporal, lag, rolling, and demand features
│   ├── forecasting/              # sktime estimators, baselines, and global models
│   ├── hierarchy/                # Bottom-up, top-down, MinT/OLS reconciliation
│   ├── evaluation/               # Metrics (WRMSSE, WAPE, sMAPE) and backtesting
│   ├── inventory/                # Lead-time demand, safety stock, and reorder points
│   ├── monitoring/               # Drift detection and performance tracking
│   └── utils/                    # Config, logging, reproducibility
├── tests/                        # Comprehensive Pytest test suites
├── scripts/                      # CLI entrypoints for ingestion and benchmarking
├── requirements.txt              # Pinned Python dependencies
└── main.py                       # Project entrypoint
```

---

## 2. Phase 1: Data Engineering Pipeline

The Phase 1 pipeline implements rigorous preprocessing and statistical auditing for the M5 Walmart Forecasting dataset:

- **Schema Validation**: Verifies presence and formatting of required columns across `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv`.
- **Wide-to-Long Reshaping**: Melts daily matrix `d_1...d_1913` into standard longitudinal temporal records.
- **Calendar & Pricing Join**: Maps weekly prices (`wm_yr_wk`) and calendar attributes (`snap_CA`, `snap_TX`, `snap_WI`, event indicators).
- **Distinguishing Zero Demand vs. Unlisted Status**: Prevents distortion of zero-demand metrics by separating genuine zero demand from pre-launch periods where items were not listed in stores.
- **Data Quality Audit**: Validates temporal continuity, flags negative quantities, detects duplicate temporal records, and enforces tree hierarchy properties.
- **Memory Optimization**: Downcasts numerical values and casts categorical columns, reducing memory footprint by >80%.
- **Summing Matrix ($S$)**: Automatically constructs the sparse hierarchical summing matrix $S$ connecting bottom-level SKU-store demand to all 12 M5 aggregation levels.

---

## 3. Dataset Setup

Place the three uncompressed M5 competition CSV files into `data/raw/`:
1. `sales_train_validation.csv`
2. `calendar.csv`
3. `sell_prices.csv`

Dataset URL: [Kaggle M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy/data)

### Running Demo Mode (No Download Required)
For local development, automated testing, or demonstration when the full raw dataset is not present, you can generate a schema-perfect synthetic dataset:

```bash
python scripts/prepare_data.py --demo --n-items 20 --n-stores 3 --n-days 180
```

---

## 4. Execution & Testing

### Run Data Preprocessing
```bash
python scripts/prepare_data.py
```

### Run Pytest Suite (All 150 Tests)
```bash
python -m pytest tests/ -v
```

---

## 5. Phase 12: Production REST API & Executive Dashboard

MarketPulse provides an enterprise-grade RESTful API service (FastAPI) and an interactive command center dashboard (React, Tailwind CSS, Plotly.js).

### Starting the Server
```bash
python scripts/run_server.py --host 127.0.0.1 --port 8000
```

- **Interactive Command Center**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive OpenAPI Documentation (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### API Endpoints
| Route | Method | Description |
|---|---|---|
| `/api/v1/health` | GET | System operational health, version, and server timestamp. |
| `/api/v1/metadata` | GET | M5 12-level hierarchy layout, available series, and models. |
| `/api/v1/forecast` | POST | Point predictions ($q_{50}$) and probabilistic fan envelopes ($q_{05}$ to $q_{95}$). |
| `/api/v1/models` | GET | List available sktime forecasters (`lightgbm`, `auto_arima`, `croston`, `seasonal_naive`). |
| `/api/v1/hierarchy/reconcile` | POST | Multi-level hierarchical reconciliation (MinT-Shrinkage, Bottom-Up, OLS). |
| `/api/v1/hierarchy/levels` | GET | Structural metadata for all 12 M5 aggregation levels. |
| `/api/v1/inventory/policy` | POST | Analytical $(s, Q)$ replenishment policy optimization ($SS, ROP, EOQ, S$, annual costs). |
| `/api/v1/inventory/simulate` | POST | Discrete-event Monte Carlo inventory simulation with stochastic lead times. |
| `/api/v1/monitoring/drift` | GET | Covariate PSI drift, Kolmogorov-Smirnov statistics, and Wasserstein distances. |
| `/api/v1/monitoring/retraining-decision` | GET | Objective 4-tier model retraining governance decision gate. |
| `/api/v1/experiments/benchmarks` | GET | Research benchmark explorer summaries for Experiments 1 through 6. |

### Dashboard Modules
1. **Demand Forecasting & Prediction Intervals**: Interactive SKU series selection, model toggles, configurable horizons ($H \in \{7, 14, 28, 56\}$), and probabilistic fan charts.
2. **Hierarchical Coherence & Reconciliation**: Interactive coherence comparison (base vs MinT reconciled), Ledoit-Wolf shrinkage, and 12-level aggregation table.
3. **Inventory Replenishment & Monte Carlo Simulator**: Cost and lead-time sliders, dynamic $SS/ROP/EOQ$ scorecards, and daily discrete-event inventory trajectories.
4. **Drift Surveillance & Retraining Gate**: Feature-level PSI risk table, CUSUM change-point control alerts, and statistical retraining justification banner.
5. **Research Benchmark Explorer**: Findings, metric scorecards, and cross-model evaluations across Experiments 1–6.

