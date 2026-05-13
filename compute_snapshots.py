import json
import math
from collections import defaultdict

MARKETS_FILE = 'data/processed/clean_markets.json'
BETS_FILE = 'data/raw/bets.json'
OUTPUT = 'data/processed/dataset.json'

DAY_MS = 24 * 60 * 60 * 1000


def stream_bets(path):
    with open(path) as f:
        f.read(2)
        buf = ''
        depth = 0
        for line in f:
            for ch in line:
                if ch == '{':
                    depth += 1
                if depth > 0:
                    buf += ch
                if ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            yield json.loads(buf)
                        except json.JSONDecodeError:
                            pass
                        buf = ''


def main():
    print('Loading clean markets...')
    with open(MARKETS_FILE) as f:
        markets = json.load(f)

    market_index = {m['id']: m for m in markets}
    market_ids = set(market_index.keys())
    print(f'Markets to process: {len(market_ids)}')

    bets_by_market = defaultdict(list)

    print('Streaming bets file (this will take a few minutes)...')
    total = 0
    kept = 0
    for bet in stream_bets(BETS_FILE):
        total += 1
        if total % 5_000_000 == 0:
            print(f'  {total:,} bets processed, {kept:,} kept...')

        contract_id = bet.get('contractId')
        if contract_id not in market_ids:
            continue

        market = market_index[contract_id]
        snapshot_time = market['snapshot_time']

        if bet['createdTime'] >= snapshot_time:
            continue

        bets_by_market[contract_id].append({
            'createdTime': bet['createdTime'],
            'probAfter': bet['probAfter'],
            'amount': bet['amount'],
            'userId': bet['userId'],
        })
        kept += 1

    print(f'Done. Total bets scanned: {total:,}, kept: {kept:,}')

    print('Computing snapshot features...')
    dataset = []
    no_bets = 0

    for market in markets:
        mid = market['id']
        snapshot_time = market['snapshot_time']
        bets = sorted(bets_by_market.get(mid, []), key=lambda b: b['createdTime'])

        if not bets:
            no_bets += 1
            continue

        price_at_snapshot = bets[-1]['probAfter']
        num_traders_at_snapshot = len(set(b['userId'] for b in bets))

        volume_at_snapshot = sum(b['amount'] for b in bets if b['amount'] > 0)

        window_start = snapshot_time - DAY_MS
        volume_last_24h = sum(
            b['amount'] for b in bets
            if b['amount'] > 0 and b['createdTime'] >= window_start
        )

        lookback_time = snapshot_time - 7 * DAY_MS
        bets_before_lookback = [b for b in bets if b['createdTime'] < lookback_time]
        if bets_before_lookback:
            price_at_14d = bets_before_lookback[-1]['probAfter']
            price_change_7d = price_at_snapshot - price_at_14d
        else:
            price_change_7d = 0.0

        # bet_count_at_snapshot: total number of bets placed before snapshot
        bet_count_at_snapshot = len(bets)

        # price_volatility: std dev of probAfter across all pre-snapshot bets
        probs = [b['probAfter'] for b in bets]
        if len(probs) > 1:
            mean = sum(probs) / len(probs)
            variance = sum((p - mean) ** 2 for p in probs) / len(probs)
            price_volatility = math.sqrt(variance)
        else:
            price_volatility = 0.0

        row = {**market,
               'price_at_snapshot': price_at_snapshot,
               'num_traders_at_snapshot': num_traders_at_snapshot,
               'volume_at_snapshot': volume_at_snapshot,
               'volume_last_24h': volume_last_24h,
               'price_change_7d': price_change_7d,
               'bet_count_at_snapshot': bet_count_at_snapshot,
               'price_volatility': price_volatility}
        dataset.append(row)

    print(f'Markets with no bets before snapshot (excluded): {no_bets}')
    print(f'Final dataset: {len(dataset)} markets')

    with open(OUTPUT, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f'Saved to {OUTPUT}')


if __name__ == '__main__':
    main()
