import json
import numpy as np
from xgboost import XGBClassifier
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

FEATURE_NAMES = [
    'price_at_snapshot', 'initial_prob', 'planned_duration_days',
    'total_liquidity', 'elasticity', 'market_tier', 'total_fees',
    'last_bet_gap_days', 'num_traders_at_snapshot', 'volume_at_snapshot',
    'volume_last_24h', 'price_change_7d', 'has_category',
    'bet_count_at_snapshot', 'price_volatility', 'time_elapsed_at_snapshot',
    'bet_frequency', 'abs_price_from_half', 'question_length',
    'question_has_year', 'question_has_pct',
] + [f'cat_{c}' for c in CATEGORIES]


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

# class weight ratio for XGBoost
neg = sum(1 for y in y_train if y == 0)
pos = sum(1 for y in y_train if y == 1)
scale = neg / pos

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.8,
    scale_pos_weight=scale,
    random_state=42,
    eval_metric='logloss',
    verbosity=0,
)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)

brier    = brier_score_loss(y_test, probs)
auc      = roc_auc_score(y_test, probs)
accuracy = accuracy_score(y_test, preds)

baseline_brier = brier_score_loss(y_test, [m['price_at_snapshot'] for m in test])

print('=== XGBoost ===')
print(f'  Brier score: {brier:.4f}  (baseline: {baseline_brier:.4f})  delta: {baseline_brier - brier:+.4f}')
print(f'  AUC-ROC:     {auc:.4f}')
print(f'  Accuracy:    {100*accuracy:.1f}%')

importances = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
print('\nTop 10 features by importance:')
for name, imp in importances[:10]:
    print(f'  {imp:.4f}  {name}')
