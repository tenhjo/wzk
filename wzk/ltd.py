# Lists, Tuples, Dicts
import json
from collections.abc import Iterable
from collections import namedtuple
import copy

import numpy as np


class CopyableObject(object):
    __slots__ = ()

    def copy(self):
        return copy.copy(self)


def obj2dict(o) -> dict:
    d = {}
    for attr in o.__slots__:
        d[attr] = getattr(o, attr)

    return d


def dict2obj(d: dict, o):
    for attr in d:
        setattr(o, attr, d[attr])

    return d


def nesteddict2namedtuple(name, d):

    values, keys = [], []
    for k in d:
        keys.append(k)

        v = d[k]
        if isinstance(v, dict):
            values.append(nesteddict2namedtuple(k, v))
        else:
            values.append(v)

    T = namedtuple(name, keys)
    return T(*values)


class AttrDict(dict):
    """ Dictionary subclass whose entries can be accessed by attributes
        (as well as normally).
        https://stackoverflow.com/a/59977999/7570817
    """
    def __init__(self, *args, **kwargs):
        def from_nested_dict(data):
            """ Construct nested AttrDicts from nested dictionaries. """
            if not isinstance(data, dict):
                return data
            else:
                return AttrDict({key_: from_nested_dict(data[key_]) for key_ in data})

        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

        for key in self.keys():
            self[key] = from_nested_dict(self[key])

    def copy(self):
        return AttrDict(super().copy())

    def deepcopy(self):
        return AttrDict(copy.deepcopy(super()))

    def to_dict(self):
        return self.__dict__


def compare_dicts(a, b):
    if a.keys() != b.keys():
        return False

    for k, v in a.items():
        if isinstance(v, dict):
            if not compare_dicts(v, b[k]):
                return False
        else:
            if not np.allclose(a[k], b[k]):
                return False
    return True


def totuple(a):
    if isinstance(a, str) and len(a) == 1:
        return a
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def tolist(a):
    if isinstance(a, str) and len(a) == 1:
        return a
    if isinstance(a, np.ndarray):
        return a.tolist()
    try:
        return list(tolist(i) for i in a)
    except TypeError:
        return a


def tuple_extract(t, default, mode="default"):
    """
    default: tuple

    mode: 'default'
          'repeat'
    """

    if t is None:
        return default

    t = atleast_tuple(t)
    length_default = len(default)
    length = len(t)

    if length == length_default:
        return t

    if mode == "default":
        return t + default[length:]

    elif mode == "repeat":
        assert length == 1, "mode 'repeat' expects length(t) == 1"
        return t * length_default

    else:
        raise ValueError(f"Unknown mode '{mode}'")


def squeeze(s):
    if len(s) == 1:
        return s[0]
    else:
        return s


def remove_nones(lst):
    return [item for item in lst if item is not None]


def atleast_list(*lists, convert=True):
    """
    adapted from numpy.atleast_1d
    """
    res = []
    for lst in lists:
        if not isinstance(lst, list):
            if convert:
                lst = tolist(lst)
                if not isinstance(lst, list):
                    lst = [lst]
            else:
                lst = [lst]

        res.append(lst)

    if len(res) == 1:
        return res[0]
    else:
        return res


def atleast_tuple(*tuples, convert=True):
    """
    adapted from numpy.atleast_1d
    """
    res = []
    for tpl in tuples:
        if not isinstance(tpl, tuple):
            if convert:
                tpl = totuple(tpl)
                if not isinstance(tpl, tuple):
                    tpl = (tpl,)

            else:
                tpl = (tpl,)
        res.append(tpl)

    if len(res) == 1:
        return res[0]
    else:
        return res


def weave_lists(*args):
    # https://stackoverflow.com/a/27166171/7570817
    return [a for b in zip(*args) for a in b]


def get_indices(li, el):
    if isinstance(li, np.ndarray):
        li = li.tolist()

    indices = []
    for eli in el:
        indices.append(li.index(eli))
    return indices


def el_add(a, b):
    """
    Element-wise addition for tuples or lists.
    """

    lst = [aa + bb for aa, bb in zip(a, b)]

    if isinstance(a, list):
        return lst
    elif isinstance(a, tuple):
        return tuple(lst)
    else:
        raise ValueError(f"Unknown type {type(a)}")


def el_shift(a, shift):

    for i, aa in enumerate(a):
        if isinstance(aa, list):
            a[i] = el_shift(a=aa, shift=shift)
        else:
            a[i] += shift
    return a


def depth(a):
    return isinstance(a, (tuple, list, np.ndarray)) and max(map(depth, a)) + 1


def depth_list(lst):
    return isinstance(lst, list) and max(map(depth_list, lst)) + 1


def depth_tuple(tpl):
    return isinstance(tpl, tuple) and max(map(depth_tuple, tpl)) + 1


def flatten_gen(lst, __cur_depth=0, max_depth=100):
    for el in lst:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            __cur_depth += 1
            if __cur_depth <= max_depth:
                yield from flatten_gen(el, __cur_depth=__cur_depth, max_depth=max_depth)
            else:
                yield el
            __cur_depth -= 1

        else:
            yield el


def flatten(lst, max_depth=100):
    return list(flatten_gen(lst=lst, max_depth=max_depth))


def element_at_depth_gen(lst, d=0, with_index=False, __cur_d=0):

    def __yield1(ii, ele):
        if with_index:
            return (ii,), ele
        else:
            return el

    def __yield2(ii, ele):
        if with_index:
            return ((ii,) + ele[0]), ele[1]
        else:
            return el

    for i, el in enumerate(lst):

        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            __cur_d += 1
            if __cur_d < d:
                # better yield from ...
                for el2 in element_at_depth_gen(el, d=d, with_index=with_index, __cur_d=__cur_d):
                    yield __yield2(i, el2)

            else:  # __cur_depth == depth
                yield __yield1(i, el)
            __cur_d -= 1

        else:
            if __cur_d == depth:
                yield __yield1(i, el)


def element_at_depth(lst, d=0, with_index=False):
    return list(element_at_depth_gen(lst=lst, d=d, with_index=with_index))


def change_tuple_order(tpl):
    return tuple(map(lambda *tt: tuple(tt), *tpl))


def change_list_order(lst):
    return list(map(lambda *ll: list(ll), *lst))


def list_of_dicts2dict_of_lists(d):
    return {k: np.array([di[k] for di in d]) for k in d[0]}


def get_first_non_empty(lst):
    """
    lst = [[], [], 1, [2, 3, 4], [], []] -> 1
    lst = [[], [], False, (), [2, 3, 4], [], []] -> [2, 3, 4]
    """
    for element in lst:
        if element:
            return element


def repeat_dict(d, n):
    if d is None:
        d = {}

    d_repeat = {}

    for i in range(n):
        d_i = {}
        for key in d:
            if isinstance(d[key], (tuple, list, np.ndarray)) and len(d[key]) == n:
                d_i[key] = d[key][i]
            else:
                d_i[key] = d[key]

        d_repeat[i] = d_i
    return d_repeat


def dict_set_default(d, default):
    for k, v in default.items():
        if k not in d:
            d[k] = v

    return d


def list_allclose(a, b):
    if isinstance(a, (tuple, list)):
        return np.array([np.allclose(aa, bb) for aa, bb in zip(a, b)])
    else:
        return np.allclose(a, b)


# json
def write_dict2json(file, d, **kwargs):
    with open(file=file, mode="w") as f:
        f.write(json.dumps(d, **kwargs))


def read_json2dict(file):
    with open(file=file, mode="r") as f:
        d = json.load(f)
    return d


#  --- Dicts ----
def rename_dict_keys(d, new_keys_dict, inplace=True):

    if inplace:
        for old_k in new_keys_dict:
            d[new_keys_dict[old_k]] = d.pop(old_k)
        return d

    else:
        d2 = {}
        for old_k in new_keys_dict:
            d2[new_keys_dict[old_k]] = d[old_k]
        return d2


def invert_dict(d):
    d2 = {v: k for k, v in d.items()}
    assert len(d2) == len(d)
    return d2
