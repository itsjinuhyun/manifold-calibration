import json

INPUT = 'data/processed/dataset.json'
TRAIN_OUTPUT = 'data/processed/train.json'
TEST_OUTPUT = 'data/processed/test.json'

SPLIT_RATIO = 0.8


def main():
    with open(INPUT) as f:
        data = json.load(f)

    data.sort(key=lambda m: m['resolution_time'])

    cutoff_idx = int(len(data) * SPLIT_RATIO)
    train = data[:cutoff_idx]
    test = data[cutoff_idx:]

    print(f'Total:  {len(data)}')
    print(f'Train:  {len(train)} ({100*len(train)/len(data):.0f}%)')
    print(f'Test:   {len(test)} ({100*len(test)/len(data):.0f}%)')
    print(f'Train YES/NO: {sum(m["outcome"] for m in train)} / {sum(1-m["outcome"] for m in train)}')
    print(f'Test  YES/NO: {sum(m["outcome"] for m in test)} / {sum(1-m["outcome"] for m in test)}')

    with open(TRAIN_OUTPUT, 'w') as f:
        json.dump(train, f, indent=2)
    with open(TEST_OUTPUT, 'w') as f:
        json.dump(test, f, indent=2)

    print(f'\nSaved to {TRAIN_OUTPUT} and {TEST_OUTPUT}')


if __name__ == '__main__':
    main()
