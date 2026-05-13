import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

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

rf = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]

# ── Trader count tier analysis ────────────────────────────────────────────────
# Hypothesis: crowd is less reliable in thin markets (few traders)
# because fewer informed participants means less information aggregated into the price.
# If RF beats the baseline in low-tier markets, our features are capturing
# something the crowd missed due to low participation.

tiers = [
    ('Low  (<20 traders)',  [i for i, m in enumerate(test) if m['num_traders_at_snapshot'] < 20]),
    ('Mid  (20–99)',        [i for i, m in enumerate(test) if 20 <= m['num_traders_at_snapshot'] < 100]),
    ('High (100+)',         [i for i, m in enumerate(test) if m['num_traders_at_snapshot'] >= 100]),
]

print('=== Layer 1 & 2: Trader Count Tier ===')
print(f'{"Tier":<22} {"N":>6} {"Baseline Brier":>15} {"RF Brier":>10} {"Delta":>8}  Verdict')
print('-' * 75)

for label, idx in tiers:
    y      = y_test[idx]
    b_prob = [test[i]['price_at_snapshot'] for i in idx]
    r_prob = rf_probs[idx]

    base_b = brier_score_loss(y, b_prob)
    rf_b   = brier_score_loss(y, r_prob)
    delta  = base_b - rf_b
    verdict = 'RF beats crowd' if delta > 0 else 'crowd wins'
    print(f'{label:<22} {len(idx):>6} {base_b:>15.4f} {rf_b:>10.4f} {delta:>+8.4f}  {verdict}')

# ── Feature importance within each tier ──────────────────────────────────────
# For each tier, train a separate RF on just those training markets
# and extract feature importances. This tells us which features
# are most predictive of outcomes specifically in that tier.

print('\n=== Feature Importance by Tier (top 5) ===')

tier_train_splits = [
    ('Low  (<20 traders)',  [i for i, m in enumerate(train) if m['num_traders_at_snapshot'] < 20]),
    ('Mid  (20–99)',        [i for i, m in enumerate(train) if 20 <= m['num_traders_at_snapshot'] < 100]),
    ('High (100+)',         [i for i, m in enumerate(train) if m['num_traders_at_snapshot'] >= 100]),
]

for label, idx in tier_train_splits:
    X_sub = X_train[idx]
    y_sub = y_train[idx]
    rf_sub = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_sub.fit(X_sub, y_sub)
    importances = sorted(zip(FEATURE_NAMES, rf_sub.feature_importances_), key=lambda x: -x[1])
    print(f'\n  {label} (n={len(idx)}):')
    for name, imp in importances[:5]:
        print(f'    {imp:.4f}  {name}')


# ── Price extremity analysis ──────────────────────────────────────────────────
# Hypothesis: favorite-longshot bias — crowd overprices longshots (prices near 0)
# and underprices favorites (prices near 1). Markets near 0.5 are hardest to call.
# We check: at each price bin, does the actual YES rate match the crowd price?
# Systematic gap = miscalibration. RF should correct it if abs_price_from_half
# captures the signal.

print('\n\n=== Layer 1 & 2: Price Extremity ===')
print('Crowd price vs actual YES rate — gap shows systematic bias')
print()
print(f'{"Price bin":<12} {"N":>6} {"Crowd price":>12} {"Actual YES%":>12} {"Crowd bias":>12} {"Baseline B":>11} {"RF Brier":>9} {"Delta":>8}')
print('-' * 90)

bins = [
    ('0.0–0.2',  [i for i, m in enumerate(test) if m['price_at_snapshot'] < 0.2]),
    ('0.2–0.4',  [i for i, m in enumerate(test) if 0.2 <= m['price_at_snapshot'] < 0.4]),
    ('0.4–0.6',  [i for i, m in enumerate(test) if 0.4 <= m['price_at_snapshot'] < 0.6]),
    ('0.6–0.8',  [i for i, m in enumerate(test) if 0.6 <= m['price_at_snapshot'] < 0.8]),
    ('0.8–1.0',  [i for i, m in enumerate(test) if m['price_at_snapshot'] >= 0.8]),
]

for label, idx in bins:
    if len(idx) < 20:
        continue
    y       = y_test[idx]
    b_prob  = np.array([test[i]['price_at_snapshot'] for i in idx])
    r_prob  = rf_probs[idx]

    avg_price  = b_prob.mean()
    yes_rate   = y.mean()
    crowd_bias = avg_price - yes_rate   # positive = overconfident, negative = underconfident
    base_b     = brier_score_loss(y, b_prob)
    rf_b       = brier_score_loss(y, r_prob)
    delta      = base_b - rf_b
    bias_label = 'OVER' if crowd_bias > 0.02 else ('UNDER' if crowd_bias < -0.02 else 'OK')

    print(f'{label:<12} {len(idx):>6} {avg_price:>12.3f} {100*yes_rate:>11.1f}% {crowd_bias:>+11.3f} {base_b:>11.4f} {rf_b:>9.4f} {delta:>+8.4f}  {bias_label}')


# ── Momentum analysis ─────────────────────────────────────────────────────────
# Hypothesis: markets where the price moved sharply in the 7 days before snapshot
# (large price_change_7d) may be less reliable — the crowd is still updating
# and hasn't settled on a stable estimate.
# Positive price_change = price rising (more YES bets recently)
# Negative price_change = price falling (more NO bets recently)

print('\n\n=== Layer 1 & 2: Momentum (price_change_7d) ===')
print('Does a rapidly moving price indicate an unreliable crowd estimate?')
print()
print(f'{"Momentum bin":<25} {"N":>6} {"Avg change":>11} {"Actual YES%":>12} {"Crowd bias":>12} {"Baseline B":>11} {"RF Brier":>9} {"Delta":>8}')
print('-' * 98)

momentum_bins = [
    ('Strong down  (<-0.15)',  [i for i, m in enumerate(test) if m['price_change_7d'] < -0.15]),
    ('Mild down  (-0.15–0.0)', [i for i, m in enumerate(test) if -0.15 <= m['price_change_7d'] < 0.0]),
    ('Stable  (0.0)',          [i for i, m in enumerate(test) if m['price_change_7d'] == 0.0]),
    ('Mild up  (0.0–0.15)',    [i for i, m in enumerate(test) if 0.0 < m['price_change_7d'] <= 0.15]),
    ('Strong up  (>0.15)',     [i for i, m in enumerate(test) if m['price_change_7d'] > 0.15]),
]

for label, idx in momentum_bins:
    if len(idx) < 20:
        continue
    y       = y_test[idx]
    b_prob  = np.array([test[i]['price_at_snapshot'] for i in idx])
    r_prob  = rf_probs[idx]
    changes = np.array([test[i]['price_change_7d'] for i in idx])

    avg_change = changes.mean()
    yes_rate   = y.mean()
    avg_price  = b_prob.mean()
    crowd_bias = avg_price - yes_rate
    base_b     = brier_score_loss(y, b_prob)
    rf_b       = brier_score_loss(y, r_prob)
    delta      = base_b - rf_b

    print(f'{label:<25} {len(idx):>6} {avg_change:>+11.3f} {100*yes_rate:>11.1f}% {crowd_bias:>+11.3f} {base_b:>11.4f} {rf_b:>9.4f} {delta:>+8.4f}')


# ── Category analysis ─────────────────────────────────────────────────────────
# Hypothesis: some topics are harder to predict than others.
# Personal goals, niche categories may have more miscalibration.
# We look at both baseline Brier (Layer 1) and RF delta (Layer 2).

print('\n\n=== Layer 1 & 2: Category ===')
print(f'{"Category":<25} {"N":>6} {"Avg price":>10} {"Actual YES%":>12} {"Crowd bias":>12} {"Baseline B":>11} {"RF Brier":>9} {"Delta":>8}')
print('-' * 98)

cats = sorted(set(m['category'] for m in test))
rows = []
for cat in cats:
    idx = [i for i, m in enumerate(test) if m['category'] == cat]
    if len(idx) < 20:
        continue
    y       = y_test[idx]
    b_prob  = np.array([test[i]['price_at_snapshot'] for i in idx])
    r_prob  = rf_probs[idx]

    avg_price  = b_prob.mean()
    yes_rate   = y.mean()
    crowd_bias = avg_price - yes_rate
    base_b     = brier_score_loss(y, b_prob)
    rf_b       = brier_score_loss(y, r_prob)
    delta      = base_b - rf_b
    rows.append((cat, len(idx), avg_price, yes_rate, crowd_bias, base_b, rf_b, delta))

# sort by baseline Brier descending — worst crowd calibration first
rows.sort(key=lambda r: -r[5])
for cat, n, avg_price, yes_rate, crowd_bias, base_b, rf_b, delta in rows:
    verdict = ' ◀ RF wins' if delta > 0 else ''
    print(f'{cat:<25} {n:>6} {avg_price:>10.3f} {100*yes_rate:>11.1f}% {crowd_bias:>+11.3f} {base_b:>11.4f} {rf_b:>9.4f} {delta:>+8.4f}{verdict}')


# ── Market age analysis ───────────────────────────────────────────────────────
# Hypothesis: older markets (open for months) have had more time for informed
# traders to correct mispricing → better calibration at snapshot.
# Younger markets may still have stale or uninformed prices.

print('\n\n=== Layer 1 & 2: Market Age at Snapshot ===')
print(f'{"Age bin":<25} {"N":>6} {"Avg age (days)":>15} {"Actual YES%":>12} {"Crowd bias":>12} {"Baseline B":>11} {"RF Brier":>9} {"Delta":>8}')
print('-' * 100)

age_bins = [
    ('Young   (<30 days)',    [i for i, m in enumerate(test) if m['time_elapsed_at_snapshot'] < 30]),
    ('Mid     (30–180 days)', [i for i, m in enumerate(test) if 30 <= m['time_elapsed_at_snapshot'] < 180]),
    ('Mature  (180–365)',     [i for i, m in enumerate(test) if 180 <= m['time_elapsed_at_snapshot'] < 365]),
    ('Old     (365+ days)',   [i for i, m in enumerate(test) if m['time_elapsed_at_snapshot'] >= 365]),
]

for label, idx in age_bins:
    if len(idx) < 20:
        continue
    y       = y_test[idx]
    b_prob  = np.array([test[i]['price_at_snapshot'] for i in idx])
    r_prob  = rf_probs[idx]
    ages    = np.array([test[i]['time_elapsed_at_snapshot'] for i in idx])

    avg_age    = ages.mean()
    yes_rate   = y.mean()
    avg_price  = b_prob.mean()
    crowd_bias = avg_price - yes_rate
    base_b     = brier_score_loss(y, b_prob)
    rf_b       = brier_score_loss(y, r_prob)
    delta      = base_b - rf_b
    verdict    = ' ◀ RF wins' if delta > 0 else ''

    print(f'{label:<25} {len(idx):>6} {avg_age:>15.1f} {100*yes_rate:>11.1f}% {crowd_bias:>+11.3f} {base_b:>11.4f} {rf_b:>9.4f} {delta:>+8.4f}{verdict}')


# ── Calibration plot ──────────────────────────────────────────────────────────
# Bin markets by crowd price, compute actual YES rate per bin.
# Perfect calibration = crowd price equals actual YES rate (diagonal line).
# Deviation shows systematic over/underconfidence.
# RF overlay shows whether the model corrects or amplifies those deviations.

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print('matplotlib not installed — skipping calibration plot (pip install matplotlib)')

bins = [(i/10, (i+1)/10) for i in range(10)]
bin_labels, crowd_prices, actual_rates, rf_prices = [], [], [], []

for lo, hi in bins:
    idx = [i for i, m in enumerate(test) if lo <= m['price_at_snapshot'] < hi]
    if len(idx) < 10:
        continue
    y      = y_test[idx]
    b_prob = np.array([test[i]['price_at_snapshot'] for i in idx])
    r_prob = rf_probs[idx]

    bin_labels.append(f'{lo:.1f}–{hi:.1f}')
    crowd_prices.append(b_prob.mean())
    actual_rates.append(y.mean())
    rf_prices.append(r_prob.mean())

print(f'\n{"Bin":<10} {"Crowd price":>12} {"Actual YES%":>12} {"RF pred%":>10} {"Crowd bias":>12}')
print('-' * 60)
for i in range(len(bin_labels)):
    bias = crowd_prices[i] - actual_rates[i]
    print(f'{bin_labels[i]:<10} {crowd_prices[i]:>12.3f} {100*actual_rates[i]:>11.1f}% {100*rf_prices[i]:>9.1f}% {bias:>+11.3f}')

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration', alpha=0.5)
    ax.plot(crowd_prices, actual_rates, 'o-', color='steelblue', linewidth=2,
            markersize=7, label='Crowd price')
    ax.plot(crowd_prices, rf_prices, 's--', color='darkorange', linewidth=1.5,
            markersize=6, label='RF predicted probability', alpha=0.8)
    ax.set_xlabel('Crowd price at 7-day snapshot', fontsize=12)
    ax.set_ylabel('Actual YES rate', fontsize=12)
    ax.set_title('Calibration Plot: Crowd vs RF vs Perfect', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('calibration_plot.png', dpi=150)
    print('\nCalibration plot saved to calibration_plot.png')
