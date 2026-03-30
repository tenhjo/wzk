import numpy as np

from wzk import ltd, random2


def n2train_test(n, split=0.2):

    if isinstance(split, float) and 0 <= split <= 1:
        n_test = int(np.round(n * split))
    elif isinstance(split, int) and 0 <= split <= n:
        n_test = split
    else:
        raise ValueError(f"Unknown value for 'split': {split} ; must either be a float [0, 1] or an int [0, n_samples]")

    n_train = n - n_test
    return n_train, n_test


def train_test_split(*args, split=0.2, shuffle=False, shuffle_block_size=1, seed=None):
    """
    if split == -1, use the same set for training and testing
    If shuffle=False, than the first n_train elements are used for training and the remaining n_test for testing
    return in same order as keras / TensorFlow
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    -> more arguments: (a_train, b_train, c_train, ...), (a_test, b_test, c_test, ...)
    """

    n = np.shape(args[0])[0]
    for a in args:
        assert np.shape(a)[0] == n

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        idx = random2.block_shuffle(n, block_size=shuffle_block_size)
        args = [a[idx] for a in args]

    if split == -1:
        return ltd.change_tuple_order((a, a) for a in args)
    else:
        n_train, n_test = n2train_test(n=n, split=split)

        train_test_tuple = ltd.change_tuple_order(np.split(a, [n_train]) for a in args)
        return train_test_tuple  # ttt
