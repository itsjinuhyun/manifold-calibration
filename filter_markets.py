import json

INPUT = 'data/raw/manifold-contracts-20240706.json'
OUTPUT = 'data/processed/clean_markets.json'

MIN_TRADERS = 10
MIN_VOLUME = 100
MIN_DURATION_DAYS = 7

TIER_RANK = {'play': 0, 'basic': 1, 'plus': 2, 'premium': 3, 'crystal': 4}

TOP_CATEGORIES = {
    'sports-default', 'politics-default', 'economics-default', 'technology-default',
    'world-default', 'science-default', 'soccer', 'football', 'nfl', 'nba',
    'us-politics', 'ai', 'crypto-speculation', 'crypto-prices', 'stocks',
    'finance', 'entertainment', 'gaming', 'personal-goals', 'fun'
}


def get_category(group_slugs):
    if not group_slugs:
        return 'uncategorized'
    for slug in group_slugs:
        if slug in TOP_CATEGORIES:
            return slug
    return 'other'


def extract_features(m):
    resolution_time = m['resolutionTime']
    created_time = m['createdTime']
    close_time = m['closeTime']
    last_bet_time = m.get('lastBetTime', resolution_time)

    fees = m.get('collectedFees', {})
    total_fees = fees.get('creatorFee', 0) + fees.get('platformFee', 0) + fees.get('liquidityFee', 0)

    return {
        'id': m['id'],
        'question': m['question'],
        'outcome': 1 if m['resolution'] == 'YES' else 0,
        'resolution_time': resolution_time,
        'created_time': created_time,
        'close_time': close_time,
        'snapshot_time': resolution_time - 7 * 24 * 60 * 60 * 1000,

        # market-level features
        'initial_prob': m.get('initialProbability', 0.5),
        'planned_duration_days': (close_time - created_time) / 86400000,
        'total_liquidity': m['totalLiquidity'],
        'elasticity': m['elasticity'],
        'market_tier': TIER_RANK.get(m.get('marketTier', 'play'), 0),
        'total_fees': total_fees,
        'last_bet_gap_days': (resolution_time - last_bet_time) / 86400000,
        'total_traders': m['uniqueBettorCount'],
        'total_volume': m['volume'],
        'has_category': 1 if m.get('groupSlugs') else 0,
        'category': get_category(m.get('groupSlugs', [])),
        'group_slugs': m.get('groupSlugs', []),
    }


def main():
    print(f'Loading {INPUT}...')
    with open(INPUT) as f:
        markets = json.load(f)
    print(f'Total markets: {len(markets)}')

    filtered = []
    stats = {'non_binary': 0, 'unresolved': 0, 'bad_resolution': 0,
             'low_traders': 0, 'low_volume': 0, 'short_duration': 0, 'non_predictive': 0}

    for m in markets:
        if m.get('outcomeType') != 'BINARY':
            stats['non_binary'] += 1
            continue
        if not m.get('isResolved'):
            stats['unresolved'] += 1
            continue
        if m.get('resolution') not in ('YES', 'NO'):
            stats['bad_resolution'] += 1
            continue
        if m.get('nonPredictive'):
            stats['non_predictive'] += 1
            continue
        if m.get('uniqueBettorCount', 0) < MIN_TRADERS:
            stats['low_traders'] += 1
            continue
        if m.get('volume', 0) < MIN_VOLUME:
            stats['low_volume'] += 1
            continue
        duration_days = (m['resolutionTime'] - m['createdTime']) / 86400000
        if duration_days <= MIN_DURATION_DAYS:
            stats['short_duration'] += 1
            continue

        filtered.append(extract_features(m))

    print(f'\nFiltering breakdown:')
    for reason, count in stats.items():
        print(f'  excluded {reason}: {count}')
    print(f'\nFinal dataset: {len(filtered)} markets')
    print(f'  YES: {sum(1 for m in filtered if m["outcome"] == 1)}')
    print(f'  NO:  {sum(1 for m in filtered if m["outcome"] == 0)}')

    with open(OUTPUT, 'w') as f:
        json.dump(filtered, f, indent=2)
    print(f'\nSaved to {OUTPUT}')


if __name__ == '__main__':
    main()
