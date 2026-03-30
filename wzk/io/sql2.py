from __future__ import annotations

# from threading import Lock  # lock = Lock()
import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from typing import Literal

import numpy as np
import pandas as pd
from pandas.io import sql  # noqa

from wzk import files, ltd, strings
from wzk.logger import setup_logger
from wzk.np2 import dtypes2, numeric2object_array, object2numeric_array  # noqa

logger = setup_logger(__name__)

_CMP = "_cmp"

TYPE_TEXT = "TEXT"
TYPE_NUMERIC = "NUMERIC"
TYPE_INTEGER = "INTEGER"
TYPE_REAL = "REAL"
TYPE_BLOB = "BLOB"


def rows2sql(rows: int | list | np.ndarray, dtype: type = str, values: list | None = None) -> object:
    if isinstance(
        rows, (int, np.int8, np.int16, np.int32, np.int64, np.uint, np.uint8, np.uint16, np.uint32, np.uint64)
    ):
        if rows == -1 or rows == [-1]:
            if values is None:
                return -1
            else:
                rows = np.arange(len(values))
        else:
            rows = [int(rows)]

    elif isinstance(rows, np.ndarray) and rows.dtype == bool:
        rows = np.nonzero(rows)[0]

    assert rows is not None, rows
    rows = np.array(rows, dtype=int).reshape(-1) + 1  # Attention! Unlike in Python, SQL indices start at 1

    if dtype is str:
        return ", ".join(map(str, rows))

    elif dtype is list:
        return rows.tolist()

    else:
        raise ValueError


def columns2sql(columns: str | list[str] | None, dtype: type) -> str | list[str]:
    if columns is None:
        return "*"
    if isinstance(columns, str):
        columns = [columns]

    if dtype is str:
        return ", ".join(map(str, columns))
    elif dtype is list:
        return columns
    else:
        raise ValueError


def order2sql(order_by: str | list[str] | dict | None, dtype: type = str) -> str:
    if order_by is None:
        order_by_str = ""

    else:
        if isinstance(order_by, (str, list)):
            columns = ltd.atleast_list(order_by, convert=False)
            asc_desc = ["ASC"] * len(columns)

        elif isinstance(order_by, dict):
            columns = order_by.keys
            asc_desc = [order_by[k] for k in order_by]

        else:
            raise ValueError

        assert len(asc_desc) == len(columns)
        for ad in asc_desc:
            assert ad == "ASC" or ad == "DESC"

        order_by_str = ", ".join([f"{c} {ad}" for c, ad in zip(columns, asc_desc, strict=True)])
        order_by_str = f" ORDER BY {order_by_str}"

    if dtype is str:
        return order_by_str
    else:
        raise ValueError


@contextmanager
def open_db_connection(file: str, lock: object | None = None, close: bool = True) -> Generator[sqlite3.Connection]:
    """
    Safety wrapper for the database call.
    """
    check_same_thread = False
    isolation_level: Literal["DEFERRED"] = "DEFERRED"

    if lock is not None:
        lock.acquire()

    # TODO: pathlib — Path(file).with_suffix(".db")
    file, ext = os.path.splitext(file)
    file = f"{file}{ext or '.db'}"

    try:
        con = sqlite3.connect(database=file, check_same_thread=check_same_thread, isolation_level=isolation_level)
    except sqlite3.OperationalError as e:
        logger.debug(file)
        raise e

    try:
        yield con

    finally:
        if close:
            con.close()
        if lock is not None:
            lock.release()


def __commit(con: sqlite3.Connection) -> None:
    try:
        con.execute("COMMIT")
    except sqlite3.OperationalError:
        pass


def execute(file: str, query: str, lock: object | None = None) -> None:
    with open_db_connection(file=file, close=True, lock=lock) as con:
        # con.execute("PRAGMA max_page_count = 200000")
        # con.execute("PRAGMA page_size = 65536")
        con.execute(query)
        __commit(con=con)


def executemany(file: str, query: str, args: list, lock: object | None = None) -> None:
    with open_db_connection(file=file, close=True, lock=lock) as con:
        con.executemany(query, args)
        __commit(con=con)


def executescript(file: str, query: str, lock: object | None = None) -> None:
    with open_db_connection(file=file, close=True, lock=lock) as con:
        con.executescript(query)
        __commit(con=con)


def set_journal_mode_wal(file: str) -> None:
    # https://www.sqlite.org/pragma.html#pragma_journal_mode
    # speed up through smarter journal mode https://sqlite.org/wal.html
    execute(file=file, query="PRAGMA journal_mode=WAL")


def set_journal_mode_memory(file: str) -> None:
    execute(file=file, query="PRAGMA journal_mode=MEMORY")


def set_page_size(file: str, page_size: int = 4096) -> None:
    execute(file=file, query=f"PRAGMA page_size={page_size}")


def vacuum(file: str) -> None:
    # https://stackoverflow.com/a/23251896/7570817
    # To allow the VACUUM command to run, change the directory for temporary files to one that has enough free space.
    # assumption: that this is the case for the directory where the file itself leads
    # temp_store_directory is deprecated, but the alternatives did not work
    # TODO: pathlib — Path(file).parent
    logger.debug("vacuum %s", file)
    execute(file=file, query=f"PRAGMA temp_store_directory = '{os.path.dirname(file)}'")
    execute(file=file, query="VACUUM")


def get_tables(file: str) -> list[str]:
    with open_db_connection(file=file, close=True, lock=None) as con:
        t = pd.read_sql_query(
            sql="SELECT name FROM sqlite_master WHERE type ='table' AND name NOT LIKE 'sqlite_%'", con=con
        )
    t = t["name"].values
    return t.tolist()


def get_columns(file: str, table: str, mode: str | list[str] | None = None) -> list | pd.DataFrame:
    with open_db_connection(file=file, close=True, lock=None) as con:
        c = pd.read_sql_query(con=con, sql=f"pragma table_info({table})")

    if mode is None:
        return c

    if isinstance(mode, str):
        mode = [mode]

    res = []
    if "name" in mode:
        res.append(c.name.values.tolist())

    if "type" in mode:
        res.append(c.type.values.tolist())

    if len(res) == 1:
        res = res[0]

    return res


def summary(file: str) -> None:
    __default_width = 20

    logger.debug("summary sql-file:'%s'", file)
    tables = get_tables(file=file)
    for t in tables:
        na, ty = get_columns(file=file, table=t, mode=["name", "type"])

        w = max([len(nai) for nai in na] + [len(tyi) for tyi in ty])
        w = max(w + 3, __default_width)

        logger.debug("table: %s", t)
        logger.debug("\tcolumns: %s", " | ".join([nai.ljust(w) for nai in na]))
        logger.debug("\ttype   : %s", " | ".join([tyi.ljust(w) for tyi in ty]))
        logger.debug("\tn_rows: %s", get_n_rows(file=file, table=t))
        logger.debug("")


def rename_tables(file: str, tables: dict[str, str]) -> None:
    old_names = get_tables(file=file)
    logger.debug("rename_tables file:'%s' %s", file, tables)
    with open_db_connection(file=file, close=True, lock=None) as con:
        cur = con.cursor()
        for old in old_names:
            if old in tables:
                new = tables[old]
                cur.execute(f"ALTER TABLE `{old}` RENAME TO `{new}`")


def rename_columns(file: str, table: str, columns: dict[str, str]) -> None:
    old_list = get_columns(file=file, table=table, mode="name")
    assert isinstance(old_list, list)

    logger.debug("rename_columns file:'%s' table:'%s' %s", file, table, columns)
    with open_db_connection(file=file, close=True, lock=None) as con:
        cur = con.cursor()
        for old in columns:
            if old in old_list:
                new = columns[old]
                cur.execute(f"ALTER TABLE `{table}` RENAME COLUMN `{old}` TO `{new}`")


def get_n_rows(file: str, table: str) -> int:
    """
    Only works if the rowid's are [0, ..., i_max]
    """
    with open_db_connection(file=file, close=True, lock=None) as con:
        return pd.read_sql_query(con=con, sql=f"SELECT COALESCE(MAX(rowid), 0) FROM {table}").values[0, 0]


def integrity_check(file: str) -> str:
    with open_db_connection(file=file, close=True, lock=None) as con:
        c = pd.read_sql_query(con=con, sql="pragma integrity_check")
    logger.debug("integrity_check file:'%s' -> %s", file, c)
    return c.values[0][0]


def concatenate_tables(
    file: str, table: str, table2: str | None = None, file2: str | list[str] | None = None, lock: object | None = None
) -> None:
    if table2 is None:
        assert file2 is not None
        table2 = table

    if file2 is None:
        execute(file=file, query=f"INSERT INTO {table} SELECT * FROM {table2}", lock=lock)

    elif isinstance(file2, list):
        for f in file2:
            concatenate_tables(file=file, table=table, table2=table2, file2=f, lock=lock)

    else:
        query = f"ATTACH DATABASE '{file2}' AS filetwo; INSERT INTO {table} SELECT * FROM filetwo.{table2}"
        executescript(file=file, query=query, lock=None)


def values2bytes(value: object, column: str) -> list:
    try:
        value = np.array(value, dtype=dtypes2.str2np(column))
    except (ValueError, TypeError, KeyError) as e:
        raise ValueError(f"Error for {column}") from e

    if np.size(value[0]) > 1 and not isinstance(value[0], bytes) and not isinstance(value[0], str):
        return [xx.tobytes() for xx in value]
    else:
        return value.tolist()


def values2bytes_dict(data: dict) -> dict:
    for key in data:
        data[key] = values2bytes(value=data[key], column=key)

    return data


def bytes2values(value: object, column: str) -> object:
    # SQL saves everything in binary form -> convert back to numeric, expect the columns which are marked as cmp
    if isinstance(value[0], bytes) and not column.endswith(_CMP):
        dtype = dtypes2.str2np(s=column)
        value = np.array([np.frombuffer(v, dtype=dtype) for v in value])

    return value


def delete_tables(file: str, tables: str | list[str]) -> None:

    tables_old = get_tables(file=file)
    tables = ltd.atleast_list(tables, convert=False)
    logger.debug("delete_tables file:'%s' tables:%s", file, tables)
    for t in tables:
        assert t in tables_old, f"table {t} not in {tables_old}"
        execute(file=file, query=f"DROP TABLE {t}")
    vacuum(file=file)


def delete_rows(file: str, table: str, rows: list | np.ndarray, lock: object | None = None) -> None:
    batch_size = int(1e5)
    logger.debug("delete_rows 'file':%s table:'%s' rows:%s", file, table, rows)

    if batch_size is None or batch_size > len(rows):
        rows = rows2sql(rows, dtype=str)
        execute(file=file, lock=lock, query=f"DELETE FROM {table} WHERE ROWID in ({rows})")

    else:  # experienced some memory errors
        assert isinstance(batch_size, int)

        rows = rows2sql(rows, dtype=list)
        assert isinstance(rows, list)
        rows = np.array(rows)
        rows.sort()
        rows = rows[::-1]

        rows = np.array_split(rows, max(2, int(np.ceil(len(rows) // batch_size))))
        for r in rows:
            r = ", ".join(map(str, r.tolist()))
            execute(file=file, lock=lock, query=f"DELETE FROM {table} WHERE ROWID in ({r})")

    vacuum(file)


def delete_columns(file: str, table: str, columns: str | list[str], lock: object | None = None) -> None:
    columns = columns2sql(columns, dtype=list)
    old_columns = get_columns(file=file, table=table, mode="name")
    assert isinstance(old_columns, list)

    for col in columns:
        if col in old_columns or col == "*":
            execute(file=file, lock=lock, query=f"ALTER TABLE {table} DROP COLUMN {col}")
    vacuum(file)


def add_column(file: str, table: str, column: str, dtype: str, lock: object | None = None) -> None:
    columns = get_columns(file=file, table=table, mode="name")
    assert isinstance(columns, list)

    if column in columns:
        logger.debug("columns %s already exists", column)
    else:
        execute(file=file, query=f"ALTER TABLE {table} ADD COLUMN {column} {dtype}", lock=lock)


def copy_column(
    file: str, table: str, column_src: str, column_dst: str, dtype: str, lock: object | None = None
) -> None:
    column_list = get_columns(file, table, mode="name")
    assert isinstance(column_list, list)
    assert column_src in column_list
    if column_dst not in column_list:
        add_column(file=file, table=table, column=column_dst, dtype=dtype, lock=lock)
    execute(file=file, query=f"UPDATE {table} SET {column_dst} = CAST({column_src} as {dtype})", lock=lock)


def copy_table(
    file: str,
    table_src: str,
    table_dst: str,
    columns: list[str] | None = None,
    dtypes: list[str] | None = None,
    order_by: str | list[str] | dict | None = None,
) -> None:
    columns_old = get_columns(file=file, table=table_src, mode=None)
    if columns is None:
        columns = columns_old.name.values
    if dtypes is None:
        dtypes = columns_old.type.values

    columns = columns2sql(columns, dtype=list)
    dtypes = columns2sql(dtypes, dtype=list)
    assert len(columns) == len(dtypes)

    columns_dtype_str = ", ".join([f"{c} {d}" for c, d in zip(columns, dtypes, strict=True)])
    columns_cast_dtype_str = ", ".join([f"CAST({c} AS {d})" for c, d in zip(columns, dtypes, strict=True)])
    order_by_str = order2sql(order_by=order_by, dtype=str)

    execute(file=file, query=f"CREATE TABLE {table_dst}({columns_dtype_str})")
    execute(file=file, query=f"INSERT INTO {table_dst} SELECT {columns_cast_dtype_str} FROM {table_src} {order_by_str}")


def sort_table(file: str, table: str, order_by: str | list[str] | dict) -> None:
    logger.debug("sort_table file:'%s' table:'%s'", file, table)
    alter_table(file=file, table=table, columns=None, dtypes=None, order_by=order_by)


def alter_table(
    file: str,
    table: str,
    columns: list[str] | None,
    dtypes: list[str] | None,
    order_by: str | list[str] | dict | None = None,
) -> None:
    table_tmp = table + strings.uuid4()
    copy_table(file=file, table_src=table, table_dst=table_tmp, columns=columns, dtypes=dtypes, order_by=order_by)
    delete_tables(file, tables=table)
    rename_tables(file, tables={table_tmp: table})


def squeeze_table(file: str, table: str, log_level: int = 1) -> None:
    columns = get_columns(file=file, table=table, mode="name")

    for c in zip(columns):
        v0 = get_values(file=file, table=table, columns=c, rows=0, return_type="list")
        if np.size(v0) == 1:
            if log_level > 0:
                logger.debug(c)
            v = get_values(file=file, table=table, columns=c, return_type="list")
            v = np.squeeze(v)
            set_values(file=file, table=table, values=(v.tolist(),), columns=c)


def change_column_dtype(file: str, table: str, column: str, dtype: str, lock: object | None = None) -> None:
    column_tmp = f"{column}{strings.uuid4()}"
    copy_column(file=file, table=table, column_src=column, column_dst=column_tmp, dtype=dtype)
    delete_columns(file=file, table=table, columns=column, lock=lock)
    copy_column(file=file, table=table, column_src=column_tmp, column_dst=column, dtype=dtype)
    delete_columns(file=file, table=table, columns=column_tmp, lock=lock)


# Get and Set SQL values
def get_values(
    file: str,
    table: str,
    columns: str | list[str] | None = None,
    rows: int | list | np.ndarray = -1,
    return_type: str = "list",
    squeeze_col: bool = True,
    squeeze_row: bool = True,
) -> object:
    """
    'i_samples' == i_samples_global
    """

    lock = None  # Lock is not necessary fo reading

    columns = columns2sql(columns=columns, dtype=list)
    columns_str = columns2sql(columns=columns, dtype=str)

    rows = rows2sql(rows, dtype=str)

    if rows == -1:  # All samples
        with open_db_connection(file=file, close=True, lock=lock) as con:
            df = pd.read_sql_query(
                con=con,
                sql=f"SELECT {columns_str} FROM {table}",
                index_col=None,
            )

    else:
        with open_db_connection(file=file, close=True, lock=lock) as con:
            try:
                df = pd.read_sql_query(
                    con=con, sql=f"SELECT {columns_str} FROM {table} WHERE ROWID in ({rows})", index_col=None
                )
            except pd.io.sql.DatabaseError as e:
                logger.debug("file '%s' table '%s'", file, table)
                raise pd.io.sql.DatabaseError from e

    value_list = []
    if np.any(columns == "*"):
        columns = df.columns.values

    if return_type == "list":
        for col in columns:
            value = bytes2values(value=df.loc[:, col].values, column=col)
            value_list.append(value)

        if len(df) == 1 and squeeze_row:
            for i in range(len(columns)):
                value_list[i] = value_list[i][0]

        if len(value_list) == 1 and squeeze_col:
            value_list = value_list[0]

        return value_list

    # Return pandas.DataFrame
    elif return_type == "df" or return_type == "dict":
        for col in columns:
            value = bytes2values(value=df.loc[:, col].values, column=col)
            try:
                df.loc[:, col] = value.tolist()
            except ValueError:
                df.loc[:, col] = numeric2object_array(value)

        if return_type == "df":
            return df
        else:
            return df2dict(df=df)

    else:
        raise ValueError(f"Invalid return_type '{return_type}'")


def set_values(
    file: str,
    table: str,
    values: tuple,
    columns: str | list[str],
    rows: int | list | np.ndarray = -1,
    lock: object | None = None,
) -> None:
    """
    values = ([...], [...], [...], ...)
    """

    set_journal_mode_wal(file=file)

    rows = rows2sql(rows, values=values[0], dtype=list)
    columns = columns2sql(columns, dtype=list)
    values = tuple(values2bytes(value=v, column=c) for v, c in zip(values, columns, strict=True))

    columns = "=?, ".join(map(str, columns))
    columns += "=?"

    values_rows_sql = ltd.change_tuple_order(values + (rows,))
    values_rows_sql = list(values_rows_sql)
    query = f"UPDATE {table} SET {columns} WHERE ROWID=?"

    executemany(file=file, query=query, args=values_rows_sql, lock=lock)


def df2sql(
    df: pd.DataFrame | None,
    file: str,
    table: str,
    dtype: dict | None = None,
    if_exists: Literal["fail", "replace", "append"] = "fail",
) -> None:
    """
    From DataFrame.to_sql():
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
                   - fail: If 'table' exists, do nothing.
                   - replace: If 'table' exists, drop it, recreate it, and insert Measurements.
                   - append: If 'table' exists, insert Measurements. Create it if it does not exist.
    """
    file = files.ensure_file_extension(file=file, ext=".db")
    if df is None:
        logger.debug("No DataFrame was provided...")
        return

    elif len(df) == 0:
        logger.debug("DataFrame is empty...")
        return

    data = df.to_dict(orient="list")
    data = values2bytes_dict(data=data)
    df = pd.DataFrame(data=data)
    with open_db_connection(file=file, close=True, lock=None) as con:
        df.to_sql(name=table, con=con, if_exists=if_exists, index=False, chunksize=None, dtype=dtype)

    set_journal_mode_wal(file=file)


def df2dict(df: pd.DataFrame, squeeze: bool = True) -> dict:
    d = df.to_dict(orient="list")
    if len(df) == 1 and squeeze:
        d = {k: d[k][0] for k in d}
    return d


class Col:
    __slots__ = (
        "name",
        "shape",
        "type_np",
        "type_sql",
    )

    def __init__(self, name: str, type_sql: str, type_np: str, shape: int | tuple[int, ...]) -> None:
        self.name: str = name
        self.type_sql: str = type_sql
        self.type_np: str = type_np
        self.shape: int | tuple = shape

    def __call__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"SQL Column ({self.name} | {self.type_sql} | {self.type_np} | {self.shape})"


class Table:
    __slots__ = (
        "cols",
        "table",
    )

    def __init__(self, table: str | None = None, cols: list[Col] | None = None) -> None:
        self.table: str = table
        self.cols: list[Col] = cols

    def __call__(self) -> str:
        return self.table

    def __getitem__(self, item: int) -> Col:
        self.cols.__getitem__(item)

    def __len__(self) -> int:
        return len(self.cols)

    def __iter__(self) -> object:
        return self.cols.__iter__()

    def names(self) -> list[str]:
        return [c.name for c in self.cols]

    def types_sql(self) -> list[str]:
        return [c.type_sql for c in self.cols]

    def types_dict_sql(self) -> dict[str, str]:
        return {c.name: c.type_sql for c in self.cols}

    def types_np(self) -> list[str]:
        return [c.type_np for c in self.cols]
