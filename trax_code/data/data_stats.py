import json
import numpy as np
import matplotlib.pyplot as plt
import sys

from numpy.lib.function_base import percentile

if len(sys.argv) != 2:
    print('Usage: python data_stats.py dataset_dir_name')
    sys.exit(1)

for data_type in ['train', 'test']:
    file_path = sys.argv[1] + '/' + data_type + '/data.json'

    print('-----', file_path)
    print()
    with open(file_path, 'r') as f:
        data = f.read()

    data = [json.loads(row) for row in data.strip().split('\n')]

    flat = []
    for row in data:
        # print(len(row['target']))
        for item in row['target']:
            flat.append(item)
    flat = np.array(flat)

    print('min:', min(flat))
    print('max:', max(flat))
    percentiles = [90.0, 95.0, 99.0, 99.9]
    print('percentiles (' + str(percentiles) + ', max):')
    print(np.percentile(flat, percentiles), max(flat))
    print()

    print('rows with max value (min, non-zero min, max):')
    max_value = max(flat)
    for row in data:
        if max_value in row['target']:
            # print(row)
            print(min(row['target']), min(x for x in row['target'] if x > 0.0), max(row['target']))
    print()

    print('number of rows:', len(data))
    print()
    print('shortest row length:', min([len(row['target']) for row in data]))
    print('longest row length: ', max([len(row['target']) for row in data]))
    print()

    # plt.hist(flat, bins=200)
    # plt.show()
