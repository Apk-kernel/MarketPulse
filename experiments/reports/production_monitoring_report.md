# MarketPulse Production Monitoring, Drift Detection & Retraining Monograph

**Author:** MarketPulse ML Systems & MLOps Architecture Team  
**Generated:** November 06, 2026  
**Scope:** Automated covariate shift, residual tracking signal, CUSUM analysis, and retraining justification.

---

## Executive Summary & System Health Dashboard

Deploying hierarchical time-series models into live retail operations requires rigorous, non-stationary environmental surveillance. The MarketPulse Monitoring Engine operates an automated multi-tier surveillance loop evaluating **Covariate Shift** ($P(X) \to P'(X)$), **Concept Drift** ($P(Y|X) \to P'(Y|X)$), and **Target Distribution Dynamics** ($P(Y) \to P'(Y)$).

### Live Operational Health Scorecard

| Monitoring Window | Date Range | WAPE (%) | $\Delta$ WAPE | Tracking Signal ($TS$) | CUSUM Alarm | Max Feature PSI | Retraining Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Window 1 (Normal Operations)** | 2011-03-30 to 2011-04-12 | 12.0% | -0.0% | 0.00 | NORMAL | 0.016 | **`NO_ACTION`** |
| **Window 2 (Demand Shock)** | 2011-04-13 to 2011-04-28 | 34.82% | +190.2% | 16.00 | ALARM | 0.056 | **`CRITICAL_FALLBACK`** |

---

## 1. Covariate Shift & Feature Distribution Drift

Feature drift is evaluated using the **Population Stability Index (PSI)** with decile discretization, two-sample **Kolmogorov-Smirnov (KS)** tests for continuous features, and **Jensen-Shannon Divergence** for categorical attributes.

```math
PSI = \sum_{b=1}^{B} (Actual_b - Expected_b) \times \ln\left( \frac{Actual_b + \epsilon}{Expected_b + \epsilon} \right)
```

| feature_name   | feature_type   |    psi | psi_status   |   ks_statistic |   ks_p_value | ks_drift_detected   |   wasserstein_dist |   relative_wasserstein | js_divergence   |   sample_size_ref |   sample_size_curr | monitoring_window     |
|:---------------|:---------------|-------:|:-------------|---------------:|-------------:|:--------------------|-------------------:|-----------------------:|:----------------|------------------:|-------------------:|:----------------------|
| sell_price     | numerical      | 0.0158 | STABLE       |         0.0293 |     0.759888 | False               |             0.1013 |                 0.0318 |                 |              2700 |                630 | Window_1 (Days 61-75) |
| discount_depth | numerical      | 0.005  | STABLE       |         0.0339 |     0.588293 | False               |             0.0001 |                 0.0013 |                 |              2700 |                630 | Window_1 (Days 61-75) |
| snap_active    | numerical      | 0      | STABLE       |         0.2159 |     0        | True                |             0.2159 |                 0.451  |                 |              2700 |                630 | Window_1 (Days 61-75) |
| is_weekend     | numerical      | 0.001  | STABLE       |         0.0143 |     0.999905 | False               |             0.0143 |                 0.0312 |                 |              2700 |                630 | Window_1 (Days 61-75) |
| dayofweek      | numerical      | 0.0034 | STABLE       |         0.0143 |     0.999905 | False               |             0.0524 |                 0.0257 |                 |              2700 |                630 | Window_1 (Days 61-75) |
| sell_price     | numerical      | 0.0314 | STABLE       |         0.0299 |     0.677613 | False               |             0.1358 |                 0.0427 |                 |              2700 |                720 | Window_2 (Days 76-90) |
| discount_depth | numerical      | 0.033  | STABLE       |         0.1236 |     0        | True                |             0.0174 |                 0.3226 |                 |              2700 |                720 | Window_2 (Days 76-90) |
| snap_active    | numerical      | 0      | STABLE       |         0.1889 |     0        | True                |             0.1889 |                 0.3946 |                 |              2700 |                720 | Window_2 (Days 76-90) |
| is_weekend     | numerical      | 0.0126 | STABLE       |         0.05   |     0.112644 | False               |             0.05   |                 0.1091 |                 |              2700 |                720 | Window_2 (Days 76-90) |
| dayofweek      | numerical      | 0.0557 | STABLE       |         0.0583 |     0.040112 | True                |             0.2125 |                 0.1041 |                 |              2700 |                720 | Window_2 (Days 76-90) |

---

## 2. Target Demand Dynamics & Syntetos-Boylan Regime Shifts

Shifts in consumer purchasing velocity and transaction intermittency directly compromise model family assumptions. The system monitors zero-sales ratios ($p_0$) and Average Demand Interval ($ADI$) / Squared Coefficient of Variation ($CV^2$) transitions.

### Target Distribution Shift Summary

| Metric | Reference Baseline | Window 1 (Operational) | Window 2 (Shock) | Shift Implication |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Daily Demand** | 2.24 units | 2.11 (-5.7%) | 2.30 (+2.6%) | Volume expansion |
| **Zero-Demand Ratio ($p_0$)** | 33.4% | 36.2% | 32.5% | Increased transaction continuity |
| **Target PSI** | — | 0.0085 (`STABLE`) | 0.0146 (`STABLE`) | Statistical target divergence |
| **Regime Transitions** | — | 16 SKUs changed regime | 14 SKUs changed regime | Syntetos-Boylan classification shift |

---

## 3. Real-Time Residual Monitoring: Tracking Signal & CUSUM Control Charts

A well-calibrated forecasting model produces zero-mean, uncorrelated white noise residuals: $\mathbb{E}[e_t] = 0$. MarketPulse implements two complementary surveillance techniques:
1. **Tracking Signal (TS)**: Evaluates cumulative forecast error relative to mean absolute deviation:
   $$TS_t = \frac{\sum_{i=1}^t (y_i - \hat{y}_i)}{\frac{1}{t}\sum_{i=1}^t |y_i - \hat{y}_i|}$$
   Control limits of $|TS_t| > 3.5$ indicate systematic, non-random bias (persistent under- or over-forecasting).
2. **Tabular CUSUM Control Chart**: Detects small, persistent mean shifts in standardized residuals:
   $$S_t^+ = \max(0, S_{t-1}^+ + z_t - k), \quad S_t^- = \max(0, S_{t-1}^- - z_t - k)$$
   Alarm is triggered when $\max(S_t^+, S_t^-) > h = 4.0$.

| window_id             |   n_obs |   WAPE |   MAE |   RMSE |   delta_wape_pct |   mean_residual |   std_residual |   max_tracking_signal |   latest_tracking_signal | cusum_alarm   |   cusum_change_point |
|:----------------------|--------:|-------:|------:|-------:|-----------------:|----------------:|---------------:|----------------------:|-------------------------:|:--------------|---------------------:|
| Window_1 (Days 61-75) |     630 |  12    | 0.254 |  0.389 |            -0.01 |           0     |          0.389 |                  4.68 |                        0 | False         |                  nan |
| Window_2 (Days 76-90) |     720 |  34.82 | 0.801 |  1.11  |           190.19 |           0.598 |          0.935 |                 16    |                       16 | True          |                    2 |

---

## 4. Retraining Justification Matrix & Decision Policy

Automated retraining without empirical justification introduces severe production hazards: unnecessary GPU/CPU compute, model instability (variance churn), and potential overfitting to temporary transient noise.

MarketPulse enforces an objective, four-tier Retraining Decision Policy:

1. **`NO_ACTION`**: Operational error is consistent with validation baseline ($\Delta WAPE \le 5\%$),    Tracking Signal is centered ($|TS| \le 3.5$), and feature distributions are stable ($PSI < 0.10$).
2. **`WARNING`**: Moderate drift observed ($0.10 \le PSI < 0.25$ or $5\% < \Delta WAPE \le 15\%$), but residual    bias is not statistically significant. **Action:** Maintain current model; raise monitoring frequency.
3. **`RETRAIN_RECOMMENDED`**: Statistically significant error degradation ($\Delta WAPE > 15\%$) corroborated by    persistent systematic bias ($|TS| > 3.5$, $Z_{bias} > 3.0$) or CUSUM shift alarm. **Action:** Queue automated retraining.
4. **`CRITICAL_FALLBACK`**: Severe performance collapse ($\Delta WAPE > 50\%$). **Action:** Immediately redirect serving    traffic to robust statistical fallbacks (Seasonal Naive / Croston) while retraining.

### Evaluated Retraining Decisions

#### Window 1 Decision: `NO_ACTION` (Urgency: LOW)

- **Primary Reason:** Model performance is stable within statistical tolerance.

  - Operational WAPE (12.0%) is consistent with baseline (12.0%).
  - Residual tracking signal (|TS| = 0.0) is well within standard limits (<= 3.5).
  - Feature stability verified: Max PSI = 0.016 < 0.10.
- **Action:** Maintain current model in production; no retraining required.

#### Window 2 Decision: `CRITICAL_FALLBACK` (Urgency: CRITICAL)

- **Primary Reason:** Catastrophic error degradation: WAPE increased by 190.2%.

  - Current WAPE (34.82%) exceeds critical threshold (+50.0% over baseline 12.0%).
  - Severe persistent systematic bias: |Tracking Signal| = 16.0 > 3.5 (Z = 17.14).
- **Action:** IMMEDIATE ACTION: Switch serving traffic to robust fallback baseline (Seasonal Naive / Croston) and trigger immediate high-priority model retraining and hyperparameter re-tuning.


---

## 5. Architectural Recommendations for Enterprise MLOps Integration

1. **Streaming Metric Ingestion:** Ingest hourly transaction logs into a Kafka/Kinesis topic to compute streaming CUSUM residual metrics in real-time.
2. **Feature Store Integration:** Compute and store reference feature distribution quantiles in Feast/Hopsworks upon model registration.
3. **Automated Retraining DAG:** Trigger an Apache Airflow or Kubeflow pipeline strictly when `RETRAIN_RECOMMENDED` is returned.
4. **Circuit Breaker Protection:** Implement a gateway circuit breaker that shifts traffic to fallback baselines when `CRITICAL_FALLBACK` fires.