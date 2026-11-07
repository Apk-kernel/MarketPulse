# MarketPulse: Comprehensive Exploratory Data Analysis Report

## 1. Sales Distribution Diagnostics (Dimension A)
- **Mean Demand**: 2.23 units/day
- **Median Demand**: 2.00 units/day
- **Standard Deviation**: 2.20
- **Zero-Sales Frequency**: **33.70%** (1,365 periods)
- **Non-Zero Mean Demand**: 3.37 units/day (Std: 1.87)
- **Demand Skewness**: 0.82 (positive long-tailed distribution)

## 2. Temporal Dynamics & Seasonality (Dimension B)
### Day-of-Week Seasonality:
- **Monday**: Index = `0.947`
- **Tuesday**: Index = `1.036`
- **Wednesday**: Index = `0.946`
- **Thursday**: Index = `0.988`
- **Friday**: Index = `1.019`
- **Saturday**: Index = `1.071`
- **Sunday**: Index = `0.994`

### SNAP & Promotional Uplift:
- **SNAP Benefit Days Uplift**: `-0.78%` (Mean SNAP sales: 2.22 vs Non-SNAP: 2.24)

## 3. Product Demand Regimes: Syntetos-Boylan Framework (Dimension C)
- **Total Product-Store Series Analyzed**: 45
- **Average ADI**: 1.52 periods | **Average CV²**: 0.26
### Regime Distribution:
- **Intermittent**: `42` series (93.3%)
- **Smooth**: `3` series (6.7%)

## 4. Price Elasticity & Promotional Behavior (Dimension D)
- **Price Range**: $1.29 - $11.89 (Average: $6.14)
### Empirical Elasticity by Category (log Q / log P):
- **FOODS**: Elasticity = `-0.1383` ($R^2 = 0.0132$) -> *Inelastic (Staple demand)*
- **HOBBIES**: Elasticity = `-0.0926` ($R^2 = 0.0062$) -> *Inelastic (Staple demand)*
- **HOUSEHOLD**: Elasticity = `0.159` ($R^2 = 0.0139$) -> *Positive slope (Price-insensitive / Luxury anomaly)*

## 5. Hierarchical Demand Concentration (Dimension E)
- **Total Business Volume**: 9,043.0 units
### State-Level Volume Share:
- **State CA**: `23.7%`
- **State TX**: `34.4%`
- **State WI**: `41.9%`
### Category-Level Volume Share:
- **Category FOODS**: `33.6%`
- **Category HOBBIES**: `34.3%`
- **Category HOUSEHOLD**: `32.2%`