import json
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score
from xgboost import XGBClassifier

with open('data/processed/train.json') as f:
    train = json.load(f)
with open('data/processed/test.json') as f:
    test = json.load(f)

CATEGORIES = [
    'sports-default', 'politics-default', 'economics-default', 'technology-default',
    'world-default', 'science-default', 'soccer', 'football', 'nfl', 'nba',
    'us-politics', 'ai', 'crypto-speculation', 'crypto-prices', 'stocks',
    'finance', 'entertainment', 'gaming', 'personal-goals', 'fun', 'other', 'uncategorized'
]


def build_features(markets):
    rows = []
    for m in markets:
        cat_onehot = [1 if m['category'] == c else 0 for c in CATEGORIES]
        row = [
            m['price_at_snapshot'],
            m['initial_prob'],
            max(m['planned_duration_days'], 0),
            m['total_liquidity'],
            m['elasticity'],
            m['market_tier'],
            m['total_fees'],
            max(m['last_bet_gap_days'], 0),
            m['num_traders_at_snapshot'],
            m['volume_at_snapshot'],
            m['volume_last_24h'],
            m['price_change_7d'],
            m['has_category'],
            m['bet_count_at_snapshot'],
            m['price_volatility'],
            m['time_elapsed_at_snapshot'],
            m['bet_frequency'],
            m['abs_price_from_half'],
            m['question_length'],
            m['question_has_year'],
            m['question_has_pct'],
        ] + cat_onehot
        rows.append(row)
    return np.array(rows)


X_train = build_features(train)
y_train = np.array([m['outcome'] for m in train])
X_test  = build_features(test)
y_test  = np.array([m['outcome'] for m in test])

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

baseline_brier = brier_score_loss(y_test, [m['price_at_snapshot'] for m in test])
print(f'Baseline Brier: {baseline_brier:.4f}\n')

# Time-series CV: folds are ordered, so later folds never appear in earlier training sets
tscv = TimeSeriesSplit(n_splits=5)


def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    probs = model.predict_proba(X_te)[:, 1]
    preds = model.predict(X_te)
    brier    = brier_score_loss(y_te, probs)
    auc      = roc_auc_score(y_te, probs)
    accuracy = accuracy_score(y_te, preds)
    delta    = baseline_brier - brier
    print(f'  Brier: {brier:.4f}  delta vs baseline: {delta:+.4f}  AUC: {auc:.4f}  Acc: {100*accuracy:.1f}%')
    return brier, auc, accuracy


# ── Logistic Regression ───────────────────────────────────────────────────────
print('=== Logistic Regression (GridSearch over C) ===')
lr_grid = GridSearchCV(
    LogisticRegression(class_weight='balanced', max_iter=1000),
    param_grid={'C': [0.01, 0.1, 1, 10, 100]},
    scoring='neg_brier_score',
    cv=tscv, n_jobs=-1, verbose=0,
)
lr_grid.fit(X_train_scaled, y_train)
print(f'  Best params: {lr_grid.best_params_}')
evaluate('LR', lr_grid.best_estimator_, X_train_scaled, y_train, X_test_scaled, y_test)

# ── Random Forest ─────────────────────────────────────────────────────────────
print('\n=== Random Forest (RandomizedSearch) ===')
rf_param_dist = {
    'n_estimators':    [200, 500],
    'max_depth':       [None, 10, 20, 30],
    'min_samples_leaf':[1, 2, 5],
    'max_features':    ['sqrt', 0.3, 0.5],
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    param_distributions=rf_param_dist,
    n_iter=24, scoring='neg_brier_score',
    cv=tscv, random_state=42, n_jobs=-1, verbose=1,
)
rf_search.fit(X_train, y_train)
print(f'  Best params: {rf_search.best_params_}')
evaluate('RF', rf_search.best_estimator_, X_train, y_train, X_test, y_test)

# ── XGBoost ───────────────────────────────────────────────────────────────────
print('\n=== XGBoost (RandomizedSearch) ===')
neg = sum(1 for y in y_train if y == 0)
pos = sum(1 for y in y_train if y == 1)

xgb_param_dist = {
    'n_estimators':      [200, 300, 500],
    'max_depth':         [3, 4, 5, 6],
    'learning_rate':     [0.01, 0.05, 0.1],
    'subsample':         [0.7, 0.8, 1.0],
    'colsample_bytree':  [0.7, 0.8, 1.0],
}
xgb_search = RandomizedSearchCV(
    XGBClassifier(scale_pos_weight=neg/pos, random_state=42,
                  eval_metric='logloss', verbosity=0),
    param_distributions=xgb_param_dist,
    n_iter=24, scoring='neg_brier_score',
    cv=tscv, random_state=42, n_jobs=-1, verbose=1,
)
xgb_search.fit(X_train, y_train)
print(f'  Best params: {xgb_search.best_params_}')
evaluate('XGB', xgb_search.best_estimator_, X_train, y_train, X_test, y_test)

# ── MLP ───────────────────────────────────────────────────────────────────────
print('\n=== MLP (GridSearch) ===')
mlp_grid = GridSearchCV(
    MLPClassifier(activation='relu', max_iter=500,
                  early_stopping=True, validation_fraction=0.1, random_state=42),
    param_grid={
        'hidden_layer_sizes': [(64, 32), (128, 64), (64,), (128, 64, 32)],
        'alpha':              [0.0001, 0.001, 0.01],
    },
    scoring='neg_brier_score',
    cv=tscv, n_jobs=-1, verbose=1,
)
mlp_grid.fit(X_train_scaled, y_train)
print(f'  Best params: {mlp_grid.best_params_}')
evaluate('MLP', mlp_grid.best_estimator_, X_train_scaled, y_train, X_test_scaled, y_test)

print('\n=== Summary ===')
print(f'Baseline Brier: {baseline_brier:.4f}')
