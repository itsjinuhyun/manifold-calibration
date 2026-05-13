import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score

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
X_test = build_features(test)
y_test = np.array([m['outcome'] for m in test])

# Logistic regression requires scaled features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000)
model.fit(X_train_scaled, y_train)

probs = model.predict_proba(X_test_scaled)[:, 1]
preds = model.predict(X_test_scaled)

brier = brier_score_loss(y_test, probs)
auc = roc_auc_score(y_test, probs)
accuracy = accuracy_score(y_test, preds)

print('=== Logistic Regression ===')
print(f'  Brier score:  {brier:.4f}')
print(f'  AUC-ROC:      {auc:.4f}')
print(f'  Accuracy:     {100*accuracy:.1f}%')

# Baseline for comparison
prices = [m['price_at_snapshot'] for m in test]
baseline_brier = brier_score_loss(y_test, prices)
print(f'\n  Baseline Brier: {baseline_brier:.4f}')
print(f'  Improvement:    {baseline_brier - brier:+.4f}')

# Feature importance (coefficients)
feature_names = [
    'price_at_snapshot', 'initial_prob', 'planned_duration_days',
    'total_liquidity', 'elasticity', 'market_tier', 'total_fees',
    'last_bet_gap_days', 'num_traders_at_snapshot', 'volume_at_snapshot',
    'volume_last_24h', 'price_change_7d', 'has_category',
] + [f'cat_{c}' for c in CATEGORIES]

coefficients = list(zip(feature_names, model.coef_[0]))
coefficients.sort(key=lambda x: abs(x[1]), reverse=True)

print('\n  Top 10 most influential features (by coefficient magnitude):')
for name, coef in coefficients[:10]:
    print(f'    {coef:+.4f}  {name}')
