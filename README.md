# Limit Order Book Forecasting

This project studies short-horizon prediction in electronic limit order books using the FI-2010 benchmark dataset. The goal is to understand whether order book states contain predictive information about near-term mid-price movement, and whether that predictive signal can later be converted into execution-aware BUY / SELL / HOLD decisions.

## Core Questions

1. Can limit order book snapshots predict short-horizon mid-price movement?
2. How does predictive performance change across the five FI-2010 forecast horizons?
3. Do nonlinear models capture useful structure beyond linear baselines?
4. Which interpretable microstructure features, such as spread, depth, imbalance, and microprice, explain predictive signal?
5. Can model probabilities be converted into BUY / SELL / HOLD signals that survive bid-ask spread costs?

## Current Progress

- Set up a reproducible project structure with reusable source modules.
- Implemented FI-2010 data loading utilities for train/test splits and preprocessing variants.
- Explored FI-2010 dataset structure, including feature/label layout and target distributions.
- Built baseline models on the official CF_1 split:
  - Majority-class baseline
  - Logistic regression
  - HistGradientBoosting
- Ran model comparison across all five forecast horizons.
- Saved benchmark results in `experiments/`.

## Current Benchmark Findings

On the initial horizon-3 benchmark, the models achieved approximately:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Majority baseline | 0.306 | 0.156 |
| Logistic regression | 0.439 | 0.431 |
| HistGradientBoosting | 0.522 | 0.508 |

The all-horizon comparison shows that model performance varies significantly by forecast horizon. HistGradientBoosting performs strongly across several horizons, but class-wise F1 reveals that accuracy alone can hide failures on the stationary/flat class.

## Planned Components

- FI-2010 data ingestion and split handling
- Multi-horizon mid-price movement benchmarks
- Baseline and nonlinear model comparison
- Macro F1 and class-wise evaluation
- Interpretable microstructure feature engineering:
  - mid-price
  - spread
  - bid/ask depth
  - queue imbalance
  - multi-level imbalance
  - microprice
  - microprice deviation
  - rolling returns and volatility
- Feature-family ablation studies
- Probability calibration and confidence thresholding
- BUY / SELL / HOLD signal generation
- Execution-cost-aware backtesting using bid/ask prices
- Regime-wise diagnostics by spread, imbalance, volatility, and liquidity

## Repository Structure

```text
notebooks/
  01_fi2010_data_exploration.ipynb
  02_baseline_models.ipynb
  03_horizon_comparison.ipynb
  04_microstructure_features.ipynb

src/lob_forecasting/
  data.py
  features.py
  models.py
  evaluation.py
  signals.py
  backtest.py

experiments/
  baseline_results_cf1_horizon3.csv
  horizon_results_cf1.csv

reports/
  figures/