
from wzk.logger import log_print
import os
import re
import uuid
import numpy as np


abc = ["a", "b", "c", "d", "e", "f", "g",
       "h", "i", "j", "k", "l", "m", "n",
       "o", "p", "q", "r", "s", "t", "u",
       "v", "w", "x", "y", "z"]
ABC = [letter.upper() for letter in abc]

NL = "\n"

# https://en.wikipedia.org/wiki/Bracket
__brackets_round = "(", ")"
__brackets_square = "[", "]"
__brackets_curly = "{", "}"
__brackets_angle = "<", ">"
brackets_dict = {"()":  __brackets_round,
                 "round": __brackets_round,
                 "r": __brackets_round,

                 "[]": __brackets_square,
                 "square": __brackets_square,
                 "s": __brackets_square,

                 "{}": __brackets_curly,
                 "curly": __brackets_curly,
                 "c": __brackets_curly,

                 "<>": __brackets_angle,
                 "angles": __brackets_angle,
                 "a": __brackets_angle}

brackets_rir = "({i})"
brackets_sis = "[{i}]"
brackets_rijr = "({i}, {j})"
brackets_sijs = "[{i}, {j}]"
brackets_sissjs = "[{i}][{j}]"
brackets_rirrjr = "({i})({j})"


def clean_ends(s, e):
    if s[:len(e)] == e:
        s = s[len(e):]

    if s[-len(e):] == e:
        s = s[:-len(e)]
    return s


def brackets_wrapper(bracket, idx, multi=True):
    # n = len(idx)
    b = brackets_dict[bracket]

    if multi:
        c = ", "
        s = b[0] + "".join(str(idx_i) + c for idx_i in idx)[:-len(c)] + b[1]
    else:
        s = "".join(b[0] + str(idx_i) + b[1] for idx_i in idx)

    return s


def remove_non_numeric(*s, squeeze=True):
    res = [re.sub(r"\D", "", s_i) for s_i in s]

    if squeeze and len(res) == 1:
        return res[0]
    else:
        return res


def str0_to_n(s, n):
    if s is None:
        s = ["test"]
    elif isinstance(s, str):
        s = [s]

    if len(s) == n:
        return s
    else:
        return s[:-1] + [s[-1] + "_" + str(i) for i in range(n-len(s)+1)]


def split_insert_join(s, split, insert_pre="", insert_after=""):
    return split.join([insert_pre + ss + insert_after for ss in s.split(split)])


def tab_str(*args, tab=4, squeeze=True):
    res = tuple(split_insert_join(s=s, split="\n", insert_pre=" " * tab) for s in args)
    if len(res) == 1 and squeeze:
        return res[0]
    else:
        return res


def str2number(s, safe=False):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError as e:
            if safe:
                raise ValueError(e)
            else:
                return s


def str2file(directory, **kwargs):

    for key in kwargs:
        file, ext = os.path.splitext(key)

        with open(f"{directory}/{file}{ext}", "w") as f:
            f.write(kwargs[key])


def uuid4():
    """
    Universally Unique Identifier (UUID)
    128-Bit x 4-Bit alpha-numeric representation
     -> len() = 32
     """
    return uuid.uuid4().hex


def str2eval(s):
    s = s.replace("null", "None")
    try:
        s = eval(s, {"__builtins__": None}, {})
    except TypeError:
        log_print(s)
        raise TypeError
    return s


def find_one_of_n(s, n):
    b = [n_i in s for n_i in n]
    assert sum(b) == 1
    i = int(np.where(b)[0])
    return n[i]


def arr2str(a, brackets=__brackets_square, sep=", "):
    # this works only for smaller arrays, see numpy.repr
    a = repr(a)
    a = a.replace("array", "")
    a = a[1:-1]
    a = a.replace("[", brackets[0])
    a = a.replace("]", brackets[1])
    a = a.replace(" ", "")
    a = a.replace(",", sep)

    return a
