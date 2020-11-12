from sys import stdout

import numpy as np

from wzk.mathematics import get_stats


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=50):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """

    if total == 0:
        total = 1
        iteration = 0

    iteration += 1
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        stdout.write('\n')
    stdout.flush()


def print_table(rows, columns, data, min_cell_size=10, cell_format='.5f'):
    max_cell_size_c = max([len(c) for c in columns] + [min_cell_size])
    max_cell_size_r = max([len(r) for r in rows] + [min_cell_size])

    row_format = '{:>' + str(max_cell_size_r) + '}'
    header_format = row_format + ('{:>' + str(max_cell_size_c) + '}') * len(columns)
    data_format = row_format + ('{:>' + str(max_cell_size_c) + cell_format + '}') * len(columns)

    print(header_format.format('', *columns))
    for row_name, row_data in zip(rows, data):
        print(data_format.format(row_name, *row_data))


def print_dict(d, newline=True, message=None):
    if message is not None:
        print(message)

    for key in d:
        print(key, ':')
        print(d[key])

        if newline:
            print()


def print_stats(*args, names):
    stats = []
    for a in args:
        s = get_stats(a)
        stats.append([s[key] for key in s])

    cols = [key for key in s]
    print_table(rows=names, columns=cols, data=stats)
    return np.array(stats)


def print_correlation(bool_lists, names, percentage=False, dec=4):
    arr = np.zeros((len(bool_lists), len(bool_lists)))
    total = np.ones_like(bool_lists[0], dtype=bool)
    for i, b1 in enumerate(bool_lists):
        total = np.logical_and(total, b1)
        for j, b2 in enumerate(bool_lists):
            arr[i, j] = np.logical_and(b1, b2).mean()

    if dec:
        arr = np.round(arr, decimals=dec)

    # if percentage:
    #     arr = arr.astype(str) + '%'

    print_table(rows=names, columns=names, data=arr)
    print(f"Total: {total.sum()}/{len(total)} = {total.mean()}")

    return total
