# MarketPulse: Time-Series Temporal Backtesting Report
**Validation Strategy**: `Expanding Window` (3 Folds)
**Forecasting Horizon**: `14 days`

## Backtest Leaderboard (Mean ± Std Across Folds)

| model_name            |   horizon | WAPE (%)      | RMSE         | MAE          | RMSSE         | sMAPE (%)     |   Avg Fit Time (s) |
|:----------------------|----------:|:--------------|:-------------|:-------------|:--------------|:--------------|-------------------:|
| AutoARIMA             |        14 | 16.30 ± 5.89  | 9.06 ± 4.18  | 7.20 ± 3.40  | 0.764 ± 0.208 | 16.37 ± 5.96  |             0.7532 |
| Naive_Mean            |        14 | 16.30 ± 5.89  | 9.06 ± 4.18  | 7.20 ± 3.40  | 0.764 ± 0.208 | 16.37 ± 5.96  |             0.007  |
| Theta                 |        14 | 18.35 ± 4.95  | 10.05 ± 4.20 | 8.25 ± 3.70  | 0.845 ± 0.172 | 18.50 ± 5.07  |             0.0272 |
| Exponential_Smoothing |        14 | 18.39 ± 4.98  | 10.01 ± 4.15 | 8.26 ± 3.66  | 0.843 ± 0.175 | 18.57 ± 5.14  |             0.262  |
| Seasonal_Naive_sp7    |        14 | 22.14 ± 8.78  | 12.05 ± 5.20 | 9.62 ± 4.34  | 1.021 ± 0.294 | 22.36 ± 9.10  |             0.005  |
| Naive_Last            |        14 | 29.38 ± 17.47 | 14.77 ± 7.39 | 12.89 ± 6.84 | 1.243 ± 0.522 | 29.06 ± 14.41 |             0.0047 |