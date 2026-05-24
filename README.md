# Limit Order Book Forecasting with Microstructure Features

This project studies short-horizon mid-price movement prediction using the FI-2010 limit order book benchmark dataset. The goal is to understand whether order book states, engineered microstructure features, and recent market context contain predictive information about near-term price movement.

The project includes baseline model benchmarks, microstructure feature engineering, horizon-wise evaluation, and feature-family benchmarks. The next phase extends the pipeline toward probability calibration, signal generation, and execution-cost-aware evaluation.

## Research Questions

1. Can limit order book snapshots predict short-horizon mid-price movement?
2. How does predictive performance vary across FI-2010 forecast horizons?
3. Do nonlinear models capture useful structure beyond linear baselines?
4. Do engineered microstructure features add predictive value beyond the original 144 FI-2010 benchmark features?
5. Does recent order book context, such as rolling returns, volatility, spread, and imbalance, improve prediction quality?

## Dataset

The project uses the FI-2010 limit order book dataset. Each sample represents a limit order book snapshot, with labels corresponding to future mid-price movement at five forecast horizons.

Labels:

```text
1 = future mid-price moves up
2 = future mid-price remains stationary / flat
3 = future mid-price moves down
```

Two FI-2010 representations are used:

- Zscore: normalized benchmark feature representation used for baseline models.
- DecPre: decimal-preserved representation used to reconstruct interpretable bid/ask price and size relationships.

## Methodology

The pipeline compares several feature families and models across all five FI-2010 forecast horizons.

Feature Families

1. Full FI-2010 benchmark features

The original 144 normalized features from the FI-2010 Zscore representation.

2. Basic microstructure features

Interpretable current-snapshot features constructed from the top 10 levels of the order book:

- mid-price
- spread
- level-1 queue imbalance
- microprice
- microprice deviation
- bid/ask depth over 5 and 10 levels
- multi-level depth imbalance

3. Rolling-context microstructure features

Recent-history features built from the engineered microstructure quantities:

- mid-price returns over 1, 5, and 10 snapshots
- rolling volatility over 10 and 50 snapshots
- rolling spread mean and standard deviation
- rolling imbalance mean and standard deviation
- rolling microprice-deviation mean and standard deviation

Absolute price-level features were also removed in one experiment to test whether performance was driven by true relative or dynamic signals rather than price-level shortcuts.

4. Combined feature set

The original 144 FI-2010 benchmark features combined with engineered rolling-context microstructure features.

Models

The project benchmarks:

- majority-class baseline
- logistic regression
- HistGradientBoosting

Models are evaluated using accuracy, macro F1, and class-wise F1. Macro F1 is emphasized because the task is a three-class prediction problem and accuracy can hide poor performance on the stationary/flat class.

## Key Results

### Horizon-3 Baseline Results

Initial models using the original 144 FI-2010 benchmark features showed clear predictive signal beyond the majority baseline.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Majority baseline | 0.306 | 0.156 |
| Logistic regression | 0.439 | 0.431 |
| HistGradientBoosting | 0.522 | 0.508 |

### Feature-Family Comparison

The strongest feature family was the combined representation: original 144 Zscore benchmark features plus rolling-context microstructure features.

Best HistGradientBoosting macro-F1 results by forecast horizon:

| Horizon | Best Feature Set | Macro F1 |
|---|---|---:|
| 0 | Zscore + context microstructure | 0.775 |
| 1 | Zscore + context microstructure | 0.844 |
| 2 | Zscore + context microstructure | 0.756 |
| 3 | Zscore + context microstructure | 0.841 |
| 4 | Zscore + context microstructure | 0.882 |

### Main Findings

1. The FI-2010 benchmark features contain meaningful predictive signal beyond class imbalance.
2. Nonlinear models outperform linear models on stronger feature representations.
3. Static one-snapshot microstructure features are interpretable but insufficient by themselves.
4. Rolling-context features produce a large improvement, suggesting that recent mid-price dynamics, volatility, and persistent order book pressure are central to short-horizon prediction.
5. Removing absolute price-level features did not materially reduce performance, suggesting that the improvement is not primarily driven by absolute price-level shortcuts.
6. Combining benchmark features with engineered rolling-context microstructure features gives the strongest overall performance across horizons.

## Results Visualizations

### Feature-family comparison

![Feature-family macro F1](reports/figures/feature_family_macro_f1_hgb.png)

### Baseline model comparison

![Baseline macro F1](reports/figures/baseline_macro_f1_by_horizon.png)

### Class-wise F1 for best model

![Class-wise F1](reports/figures/best_model_classwise_f1.png)

## Current Status

Completed:

- FI-2010 data loading utilities
- dataset exploration
- horizon-wise baseline modeling
- DecPre-based order book reconstruction
- microstructure feature engineering
- rolling-context feature construction
- feature-family ablation across all five horizons

In progress:

- probability and confidence analysis
- BUY / SELL / HOLD signal generation
- execution-cost-aware backtesting
- regime-wise evaluation by spread, imbalance, volatility, and liquidity

## Next Steps

The next stage will evaluate whether model probabilities can be converted into reliable trading-style signals. The planned signal layer will map predicted class probabilities into BUY / SELL / HOLD decisions and evaluate them under bid-ask spread costs.

Planned analysis:

- confidence-bucket accuracy
- probability calibration diagnostics
- threshold-based signal generation
- execution-aware PnL simulation
- regime-wise signal performance

## Note on AI Assistance

AI tools were used to assist with boilerplate utility code, code organization, documentation drafting, and refactoring suggestions. Core project decisions, experiment design, model evaluation, result interpretation, and final analysis are reviewed and validated manually.
