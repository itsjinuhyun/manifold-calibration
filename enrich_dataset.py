import json
import re

DATASET = 'data/processed/dataset.json'
OUTPUT = 'data/processed/dataset.json'


def main():
    print('Loading dataset...')
    with open(DATASET) as f:
        dataset = json.load(f)

    year_re = re.compile(r'\b(20[2-9]\d|19\d\d)\b')
    pct_re  = re.compile(r'%|\bpercent\b', re.IGNORECASE)

    for m in dataset:
        elapsed = (m['snapshot_time'] - m['created_time']) / 86400000

        m['time_elapsed_at_snapshot'] = max(elapsed, 0.0)
        m['bet_frequency']            = m['bet_count_at_snapshot'] / elapsed if elapsed > 0 else 0.0
        m['abs_price_from_half']      = abs(m['price_at_snapshot'] - 0.5)

        q = m['question']
        m['question_length']   = len(q)
        m['question_has_year'] = 1 if year_re.search(q) else 0
        m['question_has_pct']  = 1 if pct_re.search(q) else 0

    with open(OUTPUT, 'w') as f:
        json.dump(dataset, f, indent=2)

    sample = dataset[0]
    new_cols = ['bet_count_at_snapshot', 'price_volatility', 'time_elapsed_at_snapshot',
                'bet_frequency', 'abs_price_from_half', 'question_length',
                'question_has_year', 'question_has_pct']
    print('New feature sample (first market):')
    for col in new_cols:
        print(f'  {col}: {sample[col]}')
    print(f'\nDataset saved: {len(dataset)} markets x {len(dataset[0])} columns')


if __name__ == '__main__':
    main()
