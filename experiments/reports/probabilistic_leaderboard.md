# MarketPulse: Probabilistic Forecasting Benchmark Report

Generated at: 2026-11-06 19:15:57

## 1. Executive Summary

Point forecasts are insufficient for supply chain decision-making because inventory safety stock, service level targets, and stockout risk depend critically on **uncertainty quantification**. MarketPulse implements multi-quantile gradient boosting, distribution-free conformal prediction intervals, and zero-inflated negative binomial models for intermittent SKUs.

## 2. Probabilistic Leaderboard

| Series Archetype                     | Model                                    |    CRPS |   Mean Pinball Loss |   PICP 90% (%) |   MPIW 90% (Units) |   Winkler Score (90%) |   Runtime (s) |
|:-------------------------------------|:-----------------------------------------|--------:|--------------------:|---------------:|-------------------:|----------------------:|--------------:|
| Smooth Corporate Demand (Total)      | Quantile LightGBM (Non-Crossing)         |  7.7632 |              3.8816 |           71.4 |              33.78 |                 79.63 |        2.3975 |
| Smooth Corporate Demand (Total)      | Conformal Intervals (Seasonal Naive)     | 10.5    |              6.3    |           71.4 |              42    |                 98.43 |        0.1533 |
| Intermittent SKU Demand (Store-Item) | Quantile LightGBM (Non-Crossing)         |  0.7789 |              0.3895 |           75   |               3    |                  7.29 |        0.8919 |
| Intermittent SKU Demand (Store-Item) | Conformal Intervals (Seasonal Naive)     |  1      |              0.6    |           96.4 |               4    |                  4.71 |        0.1307 |
| Intermittent SKU Demand (Store-Item) | Intermittent Zero-Inflated (NegBinomial) |  4.5102 |              2.2551 |          100   |              28    |                 28    |        0.0067 |

## 3. Mathematical Foundations & Metrics

- **Pinball Loss (Quantile Loss)**: Assesses calibration across quantiles $\alpha \in [0.05, \dots, 0.95]$.
- **Continuous Ranked Probability Score (CRPS)**: Strictly proper scoring rule quantifying distributional distance.
- **Prediction Interval Coverage Probability (PICP)**: Validates empirical coverage against nominal target (90%).
- **Mean Prediction Interval Width (MPIW)**: Sharpness metric quantifying prediction precision.
- **Winkler Score**: Composite metric rewarding sharp bounds while heavily penalizing out-of-bound observations.
- **Monotonic Rearrangement**: Guarantees zero quantile crossings (Chernozhukov et al. 2010).
