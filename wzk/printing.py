import os
import sys
import select
import time
import numpy as np

from wzk import np2


def quiet_mode_on():
    # cannot be changed back
    # copy.copy does not work
    # stdout_copy = sys.stdout
    sys.stdout = open(os.devnull, "w")
    #

# def quiet_mode_off():
#     sys.stdout = stdout_copy
#
#
# def toggle_quiet_mode():
#     pass


def input_timed(prompt, seconds, clear=True):
    if prompt is not None and prompt != "":
        print(prompt)

    i, o, e = select.select([sys.stdin], [], [], seconds)
    s = sys.stdin.readline().strip() if i else None

    if clear:
        clear_previous_lines()

    return s


def input_clear(prompt):
    s = input(prompt)
    clear_previous_lines()
    return s


def input_clear_loop(prompt, condition, error_prompt):
    i = 0
    while True:
        s = input_clear(prompt)

        if i > 0:
            clear_previous_lines()

        if condition(s):
            return s
        else:
            print(f"Try {i}: {error_prompt}")
        i += 1


def pre_string_suf(s: str, prefix: str = "", suffix: str = "", delimiter: str = " | ") -> str:
    if s == "":
        s = prefix
    elif prefix == "":
        s = s
    else:
        s = f"{prefix}{delimiter}{s}"

    if s == "":
        s = suffix
    elif suffix == "":
        s = s
    else:
        s = f"{s}{delimiter}{suffix}"

    return s


def get_progress_bar(i, n, prefix="", suffix="", bar="█"):
    bar = bar * i + "-" * (n - i)
    return f"\r{prefix} |{bar}| {suffix}"


def progress_bar(i, n, prefix="", suffix="", bar_length=None, verbose=1,
                 eta=False, _time=[-1.]):  # noqa: B006

    if verbose <= 0:
        return

    bar_length_max = 100
    if bar_length is None:
        bar_length = n
    bar_length = min(bar_length, bar_length_max)

    if n == 0:
        i, n = 0, 1
    i += 1
    suffix += f"({i}/{n})"
    if eta:
        if _time[0] == -1. or i == 0 or i == 1:
            _time[0] = time.time()
            suffix += " | 0s | ETA: ??s"
        else:
            dt_s = (time.time() - _time[0])
            s_per_i = dt_s/i
            eta_s = (n-i) * s_per_i
            suffix += f" | Elapsed: {dt_s:.3}s | ETA: {eta_s:.3}s"

    filled_length = int(round(bar_length * i / float(n)))
    s = get_progress_bar(i=filled_length, n=bar_length, prefix=prefix, suffix=suffix)

    sys.stdout.write(s)

    if i == n:
        sys.stdout.write("\n")

    sys.stdout.flush()


def print_table(rows, columns, data, min_cell_size=10, cell_format=".5f", padding=0):
    max_voxel_size_c = max([len(c) for c in columns] + [min_cell_size])
    max_voxel_size_r = max([len(r) for r in rows] + [min_cell_size])

    max_voxel_size_c += padding
    max_voxel_size_r += padding

    row_format = "{:>" + str(max_voxel_size_r) + "}"
    header_format = row_format + ("{:>" + str(max_voxel_size_c) + "}") * len(columns)
    data_format = row_format + ("{:>" + str(max_voxel_size_c) + "}") * len(columns)
    data_format_float = row_format + ("{:>" + str(max_voxel_size_c) + cell_format + "}") * len(columns)

    print(header_format.format("", *columns))
    for row_name, row_data in zip(rows, data):
        if isinstance(row_data[0], (int, float)):
            print(data_format_float.format(row_name, *row_data))
        else:
            print(data_format.format(row_name, *row_data))


def print_dict(d, newline=True, message=None):
    if message is not None:
        print(message)

    for key in d:
        print(key, ":")
        print(repr(d[key]))

        if newline:
            print()


def print_stats(*args, names=None, dec=4):

    if names is None:
        names = [str(i) for i in range(len(args))]
    if isinstance(names, str):
        names = [names]
    assert len(names) == len(args)

    stats = []
    s = None
    for a in args:
        s = np2.get_stats(a)
        stats.append([s[key] for key in s])

    cols = [key for key in s]

    print_table(rows=names, columns=cols, data=stats, cell_format=f".{dec}f")
    return np.array(stats)


def print_stats_bool(b, name="", dec=4):
    print(f"{name}: {np.sum(b)}/{np.size(b)} = {np.mean(b):.{dec}f}")


def print_correlation(bool_lists, names, dec=4):
    arr = np.zeros((len(bool_lists), len(bool_lists)))
    total = np.ones_like(bool_lists[0], dtype=bool)
    for i, b1 in enumerate(bool_lists):
        total = np.logical_and(total, b1)
        for j, b2 in enumerate(bool_lists):
            arr[i, j] = np.logical_and(b1, b2).mean()

    if dec:
        arr = np.round(arr, decimals=dec)

    print_table(rows=names, columns=names, data=arr)
    print(f"Total: {total.sum()}/{len(total)} = {total.mean()}")

    return total


def __wrapper_names(names, n):
    if names is None:
        names = [f"{i}" for i in range(n)]

    else:
        names = [f"{i}-{name}" for i, name in enumerate(names)]

    max_name = np.amax([len(name) for name in names])
    names = [f"{name}{' ' * (max_name - len(name))}: " for name in names]
    return names


def __x_wrapper(x):
    s = f"{x:.3f}"
    if x >= 0:
        s = "+" + s
    return s


def x_and_limits2txt(x, limits, names=None):

    # parameters
    m = 100
    bar_left = "-"
    bar_right = "-"
    bar_x = "x"

    n = len(x)
    mi, ma = limits.T
    y = (x - mi) / (ma - mi)
    y = np.round(y * m).astype(int)
    y = np.clip(a=y, a_min=1, a_max=m)

    names = __wrapper_names(names=names, n=n)
    s = ""
    for i in range(n):
        s += names[i]
        s += f"{__x_wrapper(limits[i, 0])} |"
        s += bar_left * (y[i] - 1)
        s += bar_x
        s += bar_right * (m - y[i])
        s += f"| {__x_wrapper(limits[i, 1])}"
        s += f"  ->  {__x_wrapper(x[i])}"
        s += "\n"

    return s


def print_np2rkt(a):
    s = repr(a)
    s = s.replace("array", "")
    s = s.replace("(", "")
    s = s.replace(")", "")

    s = s.replace(" ", "")
    s = s.replace(",", " ")
    s = s.replace("\n", "")

    s = s.replace("[", "#[")

    s = s.replace("]", "]\n")

    print(s)


# General Functions
# ----------------------------------------------------------------------------------------------------------------------
def print2(*args, verbose=None,
           sep=" ", end="\n", file=None, flush=False):
    v = verbose_level_wrapper(verbose=verbose)

    if v.verbose > 0:
        args = [str(a) for a in args]
        t = "\t"*v.level
        print(f"{t}{sep.join(args)}", sep=sep, end=end, file=file, flush=flush)


def print_array_3d(array_3d,
                   verbose=None):
    l, k, n = array_3d.shape
    s = [""] * k
    for ll in range(l):
        ss = repr(array_3d[ll])
        ss = ss.replace("array", f"   {ll}:")
        ss = ss.replace("(", " ")
        ss = ss.replace(")", "")
        ss = ss.split("\n")
        for kk in range(k):
            s[kk] += ss[kk]

    print2("\n".join(s), verbose=verbose)


def clear_previous_lines(n=1):
    for i in range(n):
        sys.stdout.write("\033[K")  # clear line
        sys.stdout.write("\033[F")  # back to the previous line
        sys.stdout.write("\033[K")  # clear line


def color_text(s, color, background="w", weight=0):
    """you have to print normally (black / white) once to don't have any sight effects """
    color_dict = dict(black=0, k=0,
                      red=1, r=1,
                      green=2, g=2,
                      yellow=3, y=3,
                      blue=4, b=4,
                      magenta=5, m=5,
                      cyan=6, c=6,
                      gray=7, l=7,
                      white=8, w=8)
    tc = color_dict[color.lower()]
    bc = color_dict[background.lower()]
    return f"\033[{weight};3{tc};4{bc}m{s}\033"


def verbose_level_wrapper(verbose=None, level=None):
    if isinstance(verbose, Verbosity):
        if level is not None:
            verbose.level = level

        return verbose

    elif isinstance(verbose, tuple):
        return Verbosity(verbose=verbose[0], level=verbose[0])

    else:
        if verbose is None:
            verbose = 1

        if level is None:
            level = 0

        return Verbosity(verbose=verbose, level=level)


def check_verbosity(verbose, threshold=0):
    if isinstance(verbose, int):
        return verbose > threshold
    elif isinstance(verbose, tuple):
        return verbose[0] > threshold
    else:
        raise ValueError(f"Unknown type '{type(verbose)}' for verbose")


class Verbosity:
    verbose: int
    level: int

    def __init__(self, verbose=0, level=0):
        self.verbose = verbose
        self.level = level

    def __add__(self, other):
        if isinstance(other, Verbosity):
            other = other.verbose
        res = self.copy()
        res.verbose += other
        return res

    def __sub__(self, other):
        if isinstance(other, Verbosity):
            other = other.verbose
        res = self.copy()
        res.verbose -= other
        return res

    def add_level(self, other_level):
        res = self.copy()
        res.level += other_level
        return res

    def copy(self):
        return Verbosity(verbose=self.verbose, level=self.level)

    def __repr__(self):
        return f"(verbose: {self.verbose}, level: {self.level})"


def try_verbosity():
    v10 = Verbosity(verbose=1, level=0)

    v20 = v10 + 1
    v00 = v10 - 1

    print(v10)
    print(v20)
    print(v00)


def try_clear_previous_line():
    pass
    print("A")
    print("B")
    print("C")

    clear_previous_lines()

    print("C2")
    input()


if __name__ == "__main__":
    try_clear_previous_line()
