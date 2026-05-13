# Limit Order Book Forecasting

This project studies short-horizon forecasting in limit order books using market microstructure data. The goal is to understand whether order book states and order-flow dynamics contain predictive information about near-term mid-price movement and execution risk.

The project will be developed as a research-oriented pipeline covering data loading, feature engineering, label construction, baseline modeling, model evaluation, and execution-aware backtesting.

## Core Questions

1. Can limit order book features such as spread, depth, imbalance, and microprice predict short-horizon mid-price movement?
2. Do order-flow features such as order flow imbalance improve predictive performance?
3. How do different models compare under chronological validation?
4. Do predictive signals remain useful after bid-ask spread costs and adverse selection are considered?

## Planned Components

- Limit order book data ingestion
- Mid-price, spread, depth, imbalance, and microprice features
- Order-flow and OFI-based features
- Multi-horizon mid-price movement labels
- Chronological train-validation-test splits
- Baseline and machine learning models
- Regime-wise model evaluation
- Execution-aware backtesting
- Research notes and experiment logs

## Status

Initial setup in progress.
