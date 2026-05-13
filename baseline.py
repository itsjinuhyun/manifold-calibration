import json
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score

with open('data/processed/test.json') as f:
    test = json.load(f)

actual = [m['outcome'] for m in test]
prices = [m['price_at_snapshot'] for m in test]
predictions = [1 if p > 0.5 else 0 for p in prices]

brier = brier_score_loss(actual, prices)
auc = roc_auc_score(actual, prices)
accuracy = accuracy_score(actual, predictions)

print('=== Baseline (market price at 7-day snapshot) ===')
print(f'  Brier score:  {brier:.4f}  (lower is better, 0 = perfect, 0.25 = random)')
print(f'  AUC-ROC:      {auc:.4f}  (higher is better, 1.0 = perfect, 0.5 = random)')
print(f'  Accuracy:     {100*accuracy:.1f}%')
