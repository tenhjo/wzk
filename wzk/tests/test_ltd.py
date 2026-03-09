import os
from unittest import TestCase

import numpy as np

from wzk import files, ltd

directory = f"{os.path.split(__file__)[0]}/tmp"


class Test(TestCase):

    def test_totuple(self):
        self.assertTrue(ltd.totuple("aaa") == ("a", "a", "a"))
        self.assertTrue(ltd.totuple([1, 2, [3, 4]]) == (1, 2, (3, 4)))

    def test_tolist(self):
        self.assertTrue(ltd.tolist("aaa") == ["a", "a", "a"])
        self.assertTrue(ltd.tolist((1, 2, (3, 4))) == [1, 2, [3, 4]])

    def test_tuple_extract(self):
        self.assertTrue(ltd.tuple_extract(t=(1,), default=(0, 0), mode="default") == (1, 0))
        self.assertTrue(ltd.tuple_extract(t=(1,), default=(0, 0), mode="repeat") == (1, 1))
        self.assertTrue(ltd.tuple_extract(t=(1, (3, 4)), default=(0, 0, (0, 0)), mode="default") == (1, (3, 4), (0, 0)))

    def test_safe_squeeze(self):
        self.assertTrue(ltd.squeeze(s=(1.1,)) == 1.1)
        self.assertTrue(ltd.squeeze(s=(3, 4)) == (3, 4))

    def remove_nones(self):
        self.assertTrue(ltd.remove_nones([1, None, 2, None, None, 3, None, None]) == [1, 2, 3])
        self.assertTrue(ltd.remove_nones([1, None, 2, None, None, "a", None, [None]]) == [1, 2, "a", [None]])

    def test_atleast_list(self):
        self.assertTrue(ltd.atleast_list((1, 2, 3), convert=True) == [1, 2, 3])
        self.assertTrue(ltd.atleast_list((1, 2, 3), convert=False) == [(1, 2, 3)])
        self.assertTrue(ltd.atleast_list((1, 2, (3, 4)), convert=True) == [1, 2, [3, 4]])
        self.assertTrue(ltd.atleast_list(np.array((1, 2, 3)), convert=True) == [1, 2, 3])

    def test_atleast_tuple(self):
        self.assertTrue(ltd.atleast_tuple([1, 2, 3], convert=True) == (1, 2, 3))
        self.assertTrue(ltd.atleast_tuple([1, 2, 3], convert=False) == ([1, 2, 3],))
        self.assertTrue(ltd.atleast_tuple([1, 2, [3, 4]], convert=True) == (1, 2, (3, 4)))
        self.assertTrue(ltd.atleast_tuple(np.array((1, 2, 3)), convert=True) == (1, 2, 3))

    def test_dict2json(self):
        files.mkdirs(directory)

        dummy_file = f"{directory}/dummy_dict.json"
        dict_1 = {"a": 1,
                  "b": {"aa": 2,
                        "bb": 3,
                        "cc": {"aaa": 4,
                               "bbb": 5},
                        "dd": 6},
                  "c": 7}

        ltd.write_dict2json(file=dummy_file, d=dict_1, indent=4)
        dict_2 = ltd.read_json2dict(dummy_file)
        self.assertTrue(dict_1 == dict_2)

        files.rmdirs(directory)

    def test_flatten(self):
        list_nested = [(4, (1, (2, (1, [13, 13]))))]
        list_flat = [[4, (1, (2, (1, [13, 13])))],
                     [4, 1, (2, (1, [13, 13]))],
                     [4, 1, 2, (1, [13, 13])],
                     [4, 1, 2, 1, [13, 13]],
                     [4, 1, 2, 1, 13, 13]]
        for i in range(5):
            self.assertTrue(ltd.flatten(list_nested, max_depth=i+1) == list_flat[i])

    def test_element_at_depth(self):
        test_list = [1, 2, 3, [[4], [[5]], [6, 7]], [[8, 9], [10, 11, [12, [13, 13]]]], 14, [15], [16, 17]]

        res = [((3,), [[4], [[5]], [6, 7]]), ((4,), [[8, 9], [10, 11, [12, [13, 13]]]]), ((6,), [15]), ((7,), [16, 17])]
        a = ltd.element_at_depth(test_list, d=1, with_index=True)

        self.assertTrue(res == a)

    def test_repeat_dict(self):
        d = {"a": "red",
             "b": (5, 4, 3),
             "c": True}

        res = {0: {"a": "red", "b": 5, "c": True},
               1: {"a": "red", "b": 4, "c": True},
               2: {"a": "red", "b": 3, "c": True}}
        dr = ltd.repeat_dict(d=d, n=3)
        self.assertTrue(res == dr)

    def test_weave_lists(self):
        x = [[1, 2, 3], ["a", "b", "c"], [11, 12, 13, 14]]
        res = [1, "a", 11, 2, "b", 12, 3, "c", 13]

        a = ltd.weave_lists(*x)
        self.assertTrue(a == res)

    def test_nesteddict2namedtuple(self):
        test_dict = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
        nnt = ltd.nesteddict2namedtuple("test", d=test_dict)

        self.assertTrue(nnt.a == test_dict["a"])
        self.assertTrue(nnt.b == test_dict["b"])
        self.assertTrue(nnt.c.d == test_dict["c"]["d"])
        self.assertTrue(nnt.c.e == test_dict["c"]["e"])

    def test_ObjectDict(self):
        pass
        # d = dict(a1=1, b1=2, c1=[3, 4], d1=dict(a2=11, b2=22, d2=dict(a3=111, b3=222, c3=333)))
        # o = ObjectDict(d)

    def test_list_of_dicts2dict_of_lists(self):
        d0 = [dict(a=1, b=2, c=3), dict(a=11, b=22, c=33), dict(a=111, b=222, c=333)]

        d1_true = dict(a=np.array([1, 11, 111]),
                       b=np.array([2, 22, 222]),
                       c=np.array([3, 33, 333]))

        d1 = ltd.list_of_dicts2dict_of_lists(d=d0)
        self.assertTrue(ltd.compare_dicts(d1, d1_true))
