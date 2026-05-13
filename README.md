# When Do Prediction Markets Fail?

**Research question:** Under what conditions is the Manifold market price a poor probability estimate, and can ML identify the market characteristics that predict miscalibration?

The crowd is well-calibrated on average (Brier score 0.1071). No ML model beats it globally. But subgroup analysis reveals specific, characterizable conditions where the crowd's accuracy degrades — and in one slice, a model gains.

---

## Key Findings

- **The crowd wins globally.** Random Forest achieves a Brier score of 0.1221 vs. the crowd's 0.1071. The market price is already a near-optimal probability estimate.
- **Young markets are substantially worse.** Markets under 30 days old have a baseline Brier of 0.1504 vs. 0.0636 for markets over a year old. More trading time = more price discovery.
- **Thin markets degrade but not systematically.** Low-trader markets (<20 traders) have higher error (0.1195), but there's no directional bias — just noise. ML can't correct randomness.
- **The crowd under-reacts to momentum.** Markets with sharp downward price movement are still overpriced by +0.080; sharply rising markets are underpriced by −0.074. Classic conservatism bias.
- **The only ML gain: crypto-speculation.** The crowd systematically underprices crypto events by ~14 points. Random Forest corrects this, gaining +0.0160 Brier in that slice.
- **RF failure mode is mean reversion.** RF predicts 11% where the crowd says 4% (actual: 2%), and 85% where the crowd says 96% (actual: 98%). It hedges toward 0.5 instead of trusting extreme prices.

---

## Results

| Model | Brier Score | vs. Baseline | AUC-ROC |
|---|---|---|---|
| **Baseline (crowd price)** | **0.1071** | — | 0.9289 |
| Random Forest | 0.1221 | −0.0150 | 0.9254 |
| Logistic Regression | 0.1407 | −0.0336 | 0.9193 |
| MLP (128→64→32) | 0.1565 | −0.0494 | — |
| XGBoost | 0.1663 | −0.0592 | — |
| SVM (RBF) | 0.1762 | −0.0691 | 0.8266 |

Lower Brier is better. All models use `class_weight='balanced'` for the YES/NO imbalance (38.3% / 61.7%).

---

## Pipeline

```
data/raw/
  manifold-contracts-20240706.json   ← 130,091 markets
  bets.json                          ← 6.3GB individual trades

        ↓ filter_markets.py          Step 1: binary, resolved, ≥10 traders, ≥100 mana volume
        ↓ compute_snapshots.py       Step 2: stream bets.json, compute features at resolutionTime − 7 days
        ↓ enrich_dataset.py          Step 3: add 6 derived features
        ↓ split_data.py              Step 4: time-based 80/20 split (not random)

data/processed/
  dataset.json   ← 25,275 markets × 43 features
  train.json     ← 20,220 markets (Jan 2022 – Feb 2024)
  test.json      ←  5,055 markets (Feb – Jul 2024)
```

The snapshot is fixed at `resolutionTime − 7 days`. All features are computed at this point to prevent leakage. The train/test split is time-based: random splits leak future market behavior into training folds.

---

## Quickstart

```bash
git clone https://github.com/itsjinuhyun/manifold-calibration.git && cd manifold-calibration
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run pipeline (requires raw data in data/raw/)
python filter_markets.py
python compute_snapshots.py
python enrich_dataset.py
python split_data.py

# Or skip to modeling — processed data is included
python baseline.py
python random_forest.py
python analysis.py          # all 5 subgroup slices + calibration plot
```

---

## Repo Structure

```
filter_markets.py       Step 1: filter raw markets, extract base features
compute_snapshots.py    Step 2: stream bets.json, build snapshot feature set
enrich_dataset.py       Step 3: derive time, momentum, and text features
split_data.py           Step 4: temporal train/test split

baseline.py             Crowd price as probability estimate (the benchmark)
logistic_regression.py  LR model (C=0.1, StandardScaler)
random_forest.py        RF model — best performer (n=500, sqrt features)
xgboost_model.py        XGBoost (max_depth=5, lr=0.01, subsample=0.7)
mlp_model.py            MLP (128→64→32, α=0.001, early stopping)
svm_model.py            SVM (RBF kernel, Platt scaling)
tune_models.py          Hyperparameter search with TimeSeriesSplit CV

analysis.py             Subgroup analysis — 5 slices × 2 layers + calibration plot

data/processed/         Processed dataset included (raw data not included — 7GB+)
```

---

## Dataset

25,275 binary prediction markets from [Manifold Markets](https://manifold.markets) (bulk export, July 2024). Filtered from 130,091 raw markets: binary outcome, resolved YES/NO, ≥10 traders, ≥100 mana volume, >7 day duration.

43 features: market price at snapshot, liquidity, elasticity, bet frequency, price volatility, momentum (7-day price change), market age, category (22 one-hot encoded), and more. Full column reference in `enrich_dataset.py`.

Raw data (bets.json, manifold-contracts-20240706.json) available at [Manifold's bulk data page](https://docs.manifold.markets/api#bulk-data).
