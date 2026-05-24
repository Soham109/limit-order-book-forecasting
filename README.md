# Limit Order Book Forecasting with Microstructure Features

This project studies short-horizon mid-price movement prediction using the FI-2010 limit order book benchmark dataset. It tests whether order book states, engineered market microstructure features, and recent order book context contain predictive information about near-term price movement.

The current pipeline covers FI-2010 data loading, DecPre-based order book reconstruction, microstructure feature engineering, horizon-wise model evaluation, and feature-family ablation studies. The next stage extends the project toward probability calibration, BUY / SELL / HOLD signal generation, and execution-aware evaluation under bid-ask spread costs.

## Research Questions

- Can limit order book snapshots predict short-horizon mid-price movement?
- How does prediction quality change across FI-2010 forecast horizons?
- Do nonlinear models capture structure beyond linear baselines?
- Do engineered microstructure features add signal beyond the original 144 FI-2010 benchmark features?
- Does recent order book context, such as rolling returns, volatility, spread, and imbalance, improve prediction quality?

## Dataset

The project uses the FI-2010 limit order book benchmark dataset.

Each sample represents a limit order book snapshot with labels for five forecast horizons.

| Label | Meaning |
|---:|---|
| 1 | Future mid-price moves up |
| 2 | Future mid-price remains flat / stationary |
| 3 | Future mid-price moves down |

Two FI-2010 representations are used:

- **Zscore**: normalized benchmark feature representation used for baseline modeling.
- **DecPre**: decimal-preserved representation used to reconstruct interpretable bid/ask price and size relationships.

## Methodology

### Feature Families

The project compares four feature representations:

| Feature Set | Description |
|---|---|
| **Zscore 144** | Original 144 normalized FI-2010 benchmark features |
| **Basic microstructure** | Spread, queue imbalance, microprice deviation, and 5/10-level depth pressure |
| **Rolling-context microstructure** | Rolling returns, volatility, spread statistics, imbalance persistence, and microprice-pressure persistence |
| **Combined** | Original 144 benchmark features plus rolling-context microstructure features |

### Models

The current experiments benchmark:

- Majority-class baseline
- Logistic regression
- HistGradientBoosting

Models are evaluated using:

- Accuracy
- Macro F1
- Class-wise F1 for up / flat / down labels

Macro F1 is emphasized because accuracy can hide weak performance on the stationary/flat class.

## Key Results

### Horizon-3 Baseline

Initial horizon-3 models using the original 144 FI-2010 benchmark features showed clear predictive signal beyond the majority baseline.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Majority baseline | 0.306 | 0.156 |
| Logistic regression | 0.439 | 0.431 |
| HistGradientBoosting | 0.522 | 0.508 |

### Feature-Family Ablation

The strongest feature family was the combined representation: original 144 Zscore benchmark features plus rolling-context microstructure features.

Best HistGradientBoosting macro-F1 results by forecast horizon:

| Horizon | Best Feature Set | Macro F1 |
|---:|---|---:|
| 0 | Zscore + rolling-context microstructure | 0.775 |
| 1 | Zscore + rolling-context microstructure | 0.844 |
| 2 | Zscore + rolling-context microstructure | 0.756 |
| 3 | Zscore + rolling-context microstructure | 0.841 |
| 4 | Zscore + rolling-context microstructure | 0.882 |

### Main Findings

- The original FI-2010 benchmark features contain predictive signal beyond class imbalance.
- HistGradientBoosting consistently improves over logistic regression on stronger feature representations.
- Static one-snapshot microstructure features are interpretable but weak by themselves.
- Rolling-context features substantially improve performance, suggesting that recent mid-price dynamics and persistent order book pressure are important.
- Removing absolute price-level features does not materially reduce rolling-context performance.
- Combining benchmark features with engineered rolling-context microstructure signals gives the strongest overall results.

## Results Visualizations

### Feature-Family Comparison

HistGradientBoosting performance across feature families and forecast horizons.

![Feature-family macro F1](reports/figures/feature_family_macro_f1_hgb.png)

### Baseline Model Comparison

Macro F1 across horizons using the original 144 FI-2010 benchmark features.

![Baseline macro F1](reports/figures/baseline_macro_f1_by_horizon.png)

### Class-Wise F1 for Best Model

Class-wise F1 for the best combined feature representation.

![Class-wise F1](reports/figures/best_model_classwise_f1.png)

## Current Status

Completed:

- FI-2010 data loading utilities
- Dataset exploration and label distribution analysis
- Horizon-wise baseline modeling
- DecPre-based top-10 order book reconstruction
- Microstructure feature engineering
- Rolling-context feature construction
- Feature-family ablation across five forecast horizons

In progress:

- Probability and confidence analysis
- BUY / SELL / HOLD signal generation
- Execution-cost-aware backtesting
- Regime-wise evaluation by spread, volatility, imbalance, and liquidity

## Next Steps

The next stage evaluates whether model probabilities can be converted into trading-style signals.

Planned work:

- Confidence-bucket accuracy analysis
- Probability calibration diagnostics
- Threshold-based BUY / SELL / HOLD signal generation
- Bid-ask spread cost modeling
- Execution-aware PnL simulation
- Regime-wise signal performance analysis

## Note on AI Assistance

AI tools were used to assist with boilerplate utility code, code organization, documentation drafting, and refactoring suggestions. Core project decisions, experiment design, model evaluation, result interpretation, and final analysis are reviewed and validated manually by me.