
from wzk.logger import log_print
from unittest import TestCase
import os

import numpy as np
import pandas as pd

from wzk import sql2, files


directory = f"{os.path.split(__file__)[0]}/tmp"


class Test(TestCase):

    def setUp(self):
        files.mkdirs(directory=directory)

    def tearDown(self):
        files.rmdirs(directory=directory)

    @staticmethod
    def __create_dummy_db(mode):
        if mode == "A":
            columns = ["A", "B", "C", "DD"]
            data = [[1,   2,   3.3,  4],
                    [10,  20,  33.3,  40],
                    [100, 300, 333.3, 400]]
            df = pd.DataFrame(columns=columns, data=data, index=None)
            return df
        elif mode == "B":

            columns = ["aa", "bb", "cc"]
            n = 11
            data = np.random.random((n, 3)).tolist()
            df = pd.DataFrame(columns=columns, data=data, index=None)
            return df

        elif mode == "C":
            columns = ["A", "B", "C"]
            data = [[1, 1, 33.3],
                    [1, 2, 33.3],
                    [3, 3, 33.3],
                    [1, 4, 33.3],
                    [2, 5, 33.3],
                    [1, 6, 33.3],
                    [1, 7, 33.3],
                    [4, 8, 33.3]]
            df = pd.DataFrame(columns=columns, data=data, index=None)
            return df

        elif mode == "D":
            columns = ["A_f32", "B", "C"]
            data = [[np.random.random((20, 4)), 1, 33.3],
                    [np.random.random((20, 4)), 2, 33.3],
                    [np.random.random((20, 4)), 3, 33.3],
                    [np.random.random((20, 4)), 4, 33.3],
                    [np.random.random((20, 4)), 5, 33.3],
                    [np.random.random((20, 4)), 6, 33.3],
                    [np.random.random((20, 4)), 7, 33.3],
                    [np.random.random((20, 4)), 8, 33.3]]
            df = pd.DataFrame(columns=columns, data=data, index=None)
            return df

        else:
            raise ValueError

    def test_concatenate_tables(self):

        df = self.__create_dummy_db("A")

        file1 = f"{directory}/dummy_test_concatenate_tables_1.db"
        file2 = f"{directory}/dummy_test_concatenate_tables_2.db"
        table_a = "tableA"
        table_b = "tableB"
        sql2.df2sql(df=df, file=file1, table=table_a, if_exists="replace")
        sql2.df2sql(df=df, file=file1, table=table_b, if_exists="replace")
        sql2.df2sql(df=df, file=file2, table=table_a, if_exists="replace")

        #
        sql2.concatenate_tables(file=file1, table=table_b, table2=table_a)
        df2 = sql2.get_values(file=file1, table=table_b, rows=[3, 4, 5], return_type="df")
        self.assertTrue(np.all(df == df2))

        sql2.concatenate_tables(file=file2, table=table_a, file2=file1, table2=table_b)
        df2 = sql2.get_values(file=file2, table=table_a, rows=[6, 7, 8], return_type="df")
        self.assertTrue(np.all(df == df2))
        #

    def test_set_values(self):
        file = f"{directory}/dummy_test_set_values.db"
        table = "dummytable"
        df = self.__create_dummy_db("A")
        sql2.df2sql(df=df, file=file, table=table, if_exists="replace")

        #
        c = "B"
        r = [0, 1]
        v = [-2, -20]
        sql2.set_values(file=file, table=table, values=(v,), columns=c, rows=r)
        v1 = sql2.get_values(file=file, table=table, rows=r, columns=c, return_type="list")
        self.assertTrue(np.array_equal(v, v1))

    def test_set_values_2(self):
        file = f"{directory}/dummy_test_set_values_2.db"
        table = "dummytable"
        df = self.__create_dummy_db("D")
        sql2.df2sql(df=df, file=file, table=table, if_exists="replace")

        #
        c = "A_f32"
        r = [0, 1]
        v = np.arange(2*20*4).reshape((2, 20, 4))
        # v0 = get_values(file=file, table=table, rows=r, columns=c, return_type="list")
        sql2.set_values(file=file, table=table, values=(v,), columns=c, rows=r)
        v1 = sql2.get_values(file=file, table=table, rows=r, columns=c, return_type="list")
        v1 = v1.reshape((2, 20, 4))
        self.assertTrue(np.array_equal(v, v1))

    def test_add_column(self):
        file = f"{directory}/dummy_test_set_values_3.db"
        table = "dummytable"
        df = self.__create_dummy_db("B")
        sql2.df2sql(df=df, file=file, table=table, if_exists="replace")

        #
        c = "dd"
        r = np.arange(11)
        v = np.random.random(11)
        sql2.add_column(file=file, table=table, column=c, dtype=sql2.TYPE_REAL)
        sql2.set_values(file=file, table=table, values=(v,), columns=c, rows=r)
        v1 = sql2.get_values(file=file, table=table, rows=r, columns=c, return_type="list")
        self.assertTrue(np.array_equal(v, v1))

    def test_delete_rows(self):
        file = f"{directory}/dummy_test_delete_rows.db"
        table = "dummytable"
        df = self.__create_dummy_db("A")
        sql2.df2sql(df=df, file=file, table=table, if_exists="replace")

        #
        n_rows = sql2.get_n_rows(file=file, table=table)
        self.assertTrue(n_rows == 3)

        sql2.delete_rows(file=file, table=table, rows=[0, 1])
        n_rows = sql2.get_n_rows(file=file, table=table)
        self.assertTrue(n_rows == 1)

        df2 = sql2.get_values(file=file, table=table, squeeze_row=True, squeeze_col=True)
        self.assertTrue(np.array_equal(df.iloc[2, :].values, df2))

    def test_get_columns(self):
        file = f"{directory}/dummy_test_get_columns.db"
        table = "dummytable"
        df = self.__create_dummy_db("A")
        sql2.df2sql(df=df, file=file, table=table, if_exists="replace")

        #
        c = sql2.get_columns(file=file, table=table, mode="name")
        self.assertTrue(np.array_equal(df.columns.values, c))

    def test_change_column_dtype(self):
        file = f"{directory}/dummy_test_change_column_dtype.db"
        table = "dummytable"
        df = self.__create_dummy_db("B")
        sql2.df2sql(df=df, file=file, table=table, if_exists="replace")

        #
        sql2.change_column_dtype(file, table, column="aa", dtype=sql2.TYPE_INTEGER)
        sql2.delete_columns(file=file, table=table, columns=["bb", "cc"])
        self.assertTrue(np.all(sql2.get_values(file=file, table=table, return_type="list") == 0))

    def test_alter_table(self):
        file = f"{directory}/dummy_test_alter_table.db"
        df = self.__create_dummy_db("B")
        table = "dummytable"
        sql2.df2sql(df=df, file=file, table=table, if_exists="replace")

        #
        new_columns = ["bb", "aa", "cc"]
        new_types = [sql2.TYPE_TEXT, sql2.TYPE_INTEGER, sql2.TYPE_REAL]
        sql2.alter_table(file=file, table=table, columns=new_columns, dtypes=new_types)

        data2 = sql2.get_values(file=file, table=table, return_type="list")
        bb, aa, cc = data2

        c = sql2.get_columns(file=file, table=table, mode="name")
        self.assertTrue(np.all(c == new_columns))

        self.assertTrue(isinstance(bb[0], str))
        self.assertTrue(isinstance(aa[0], np.int64))
        self.assertTrue(isinstance(cc[0], float))

    def test_sort_table(self):
        file = f"{directory}/dummy_test_sort_table.db"
        df = self.__create_dummy_db("C")
        table = "dummytable"
        sql2.df2sql(df=df, file=file, table=table, if_exists="replace")

        #
        sql2.sort_table(file=file, table=table, order_by=["A", "ROWID"])
        data2 = sql2.get_values(file=file, table=table)
        log_print(data2)
        a2 = sql2.get_values(file=file, table=table, columns="A")
        self.assertTrue(np.all(np.argsort(a2) == np.arange(len(a2))))
