# MarketPulse: Automated Model Selection & Optuna Tuning Report

**Execution Timestamp:** 2026-11-06 17:32:32 UTC
**Optimization Runtime:** 1.74s | **Selection Runtime:** 13.48s
**Forecast Horizon:** 14 days | **Cross-Validation:** 2-Fold Temporal Rolling Splits
**Frameworks:** `optuna==4.5.0`, `lightgbm`, `sktime`

## Executive Summary
Phase 9 implements an enterprise-grade automated model selection and hyperparameter optimization engine modeled after production systems at Walmart and Amazon. The engine performs:
1. **Bayesian Hyperparameter Search:** Tree-structured Parzen Estimators (TPE) with Median Pruning to optimize gradient-boosted tree architectures without lookahead leakage.
2. **Syntetos-Boylan Regime Routing:** Dynamically tailors candidate algorithms based on series intermittency and volatility.
3. **Multi-Model Tournament:** Empirically selects champion models per SKU based on out-of-fold validation.

---

## 1. Optuna Bayesian Hyperparameter Optimization
**Best Trial:** #7 | **Validation WAPE:** 12.74%

### Optimal Hyperparameter Configuration:
| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| `n_estimators` | `50` | Selected by TPE optimization |
| `learning_rate` | `0.04228762725326872` | Selected by TPE optimization |
| `num_leaves` | `50` | Selected by TPE optimization |
| `max_depth` | `6` | Selected by TPE optimization |
| `min_child_samples` | `28` | Selected by TPE optimization |
| `subsample` | `0.7888859700647797` | Selected by TPE optimization |
| `colsample_bytree` | `0.6478376983753207` | Selected by TPE optimization |
| `reg_alpha` | `0.2246548820915795` | Selected by TPE optimization |
| `reg_lambda` | `0.37575617230882064` | Selected by TPE optimization |
| `window_length` | `14` | Selected by TPE optimization |

### Hyperparameter Importance Ranking:
| Parameter | Importance Score |
| :--- | :---: |
| `learning_rate` | `0.3449` |
| `min_child_samples` | `0.2595` |
| `num_leaves` | `0.0963` |
| `window_length` | `0.0862` |
| `n_estimators` | `0.0851` |
| `subsample` | `0.0627` |
| `reg_lambda` | `0.0286` |
| `max_depth` | `0.0166` |
| `colsample_bytree` | `0.0153` |
| `reg_alpha` | `0.0047` |

---

## 2. Portfolio Model Selection Tournament
### Champion Model Market Share:
| Champion Model | Selected Series Count | Portfolio Share (%) |
| :--- | :---: | :---: |
| **Croston** | 28 | 62.2% |
| **Theta** | 8 | 17.8% |
| **Naive_Last** | 5 | 11.1% |
| **Seasonal_Naive** | 3 | 6.7% |
| **Tuned_LightGBM** | 1 | 2.2% |

### Champion Selection by Demand Regime:
| Regime       |   Croston |   Naive_Last |   Seasonal_Naive |   Theta |   Tuned_LightGBM |
|:-------------|----------:|-------------:|-----------------:|--------:|-----------------:|
| Intermittent |        28 |            5 |                2 |       7 |                0 |
| Smooth       |         0 |            0 |                1 |       1 |                1 |

### Sample SKU Model Selection Leaderboard (Top 10 Series):
| Series ID          | Regime       |   ADI |   CV2 | Champion Model   |   Champion WAPE (%) | Runner Up Model   |   Runner Up WAPE (%) |   Selection Margin |   Improvement vs Naive (%) |
|:-------------------|:-------------|------:|------:|:-----------------|--------------------:|:------------------|---------------------:|-------------------:|---------------------------:|
| CA_1_FOODS_1_000   | Intermittent |  1.5  |  0.2  | Croston          |             160.526 | Theta             |              163.388 |              2.862 |                      18.55 |
| CA_1_FOODS_1_003   | Intermittent |  1.48 |  0.24 | Croston          |              73.225 | Theta             |               84.179 |             10.954 |                      35.94 |
| CA_1_FOODS_1_006   | Intermittent |  1.58 |  0.36 | Croston          |              90.72  | Theta             |               91.724 |              1.004 |                      12.6  |
| CA_1_FOODS_1_009   | Intermittent |  1.53 |  0.36 | Croston          |              83.356 | Theta             |               94.121 |             10.765 |                      19.98 |
| CA_1_FOODS_1_012   | Intermittent |  1.58 |  0.19 | Seasonal_Naive   |              95.652 | Naive_Last        |               96.154 |              0.502 |                       0    |
| CA_1_HOBBIES_2_001 | Intermittent |  1.41 |  0.26 | Croston          |              74.044 | Theta             |               81.586 |              7.542 |                      36.61 |
| CA_1_HOBBIES_2_004 | Intermittent |  1.55 |  0.33 | Croston          |              79.54  | Theta             |               90.293 |             10.753 |                      29.8  |
| CA_1_HOBBIES_2_007 | Intermittent |  1.76 |  0.34 | Croston          |             101.847 | Theta             |              104.007 |              2.16  |                      20.55 |
| CA_1_HOBBIES_2_010 | Intermittent |  1.84 |  0.35 | Croston          |             145.004 | Theta             |              160.25  |             15.246 |                      25.02 |
| CA_1_HOBBIES_2_013 | Intermittent |  1.84 |  0.33 | Seasonal_Naive   |              90.909 | Croston           |               92.253 |              1.344 |                       0    |

**Portfolio Average Accuracy Improvement over Naive:** **+22.53%**

---

## 3. Production Deployment Guidelines
1. **Store Routing Maps:** Deploy `champion_model_routing.json` to the forecasting runner to decouple model training from daily batch inference.
2. **Continuous Pruning:** In high-frequency SKU retraining, maintain `MedianPruner` with a warmup of 1 step to save up to 40% compute resources.
3. **Regime Monitoring:** Re-run regime classification quarterly to detect transition of products from erratic to smooth demand as products mature.