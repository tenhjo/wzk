from time import time, sleep  # noqa
from datetime import datetime

from wzk.printing import print2, verbose_level_wrapper


def get_timestamp(t=None, year=True, month=True, day=True, hour=True, minute=True, second=True, millisecond=False,
                  date_separator="-", date_time_separator="_", time_separator=":"):
    """
    Crete a datetime string including year, month, day; hour, minute, second, millisecond.
    With options for each of the elements if it should be included and what the symbol separating
    two elements should look like.
    """

    if t is None:
        t = datetime.now()

    if isinstance(t, (int, float)):
        t = datetime.fromtimestamp(t)

    assert isinstance(t, datetime)

    # Create a boolean list indicating which elements to keep
    bool_list = [bool(b) for b in [year, month, day, hour, minute, second, millisecond]]

    # Create a list of symbols which separate the elements of the datetime string
    separator_list = [date_separator, date_separator,
                      date_time_separator,
                      time_separator, time_separator, time_separator]

    # Clean the datetime string and split it into separate elements
    stamp_list = str(t)
    stamp_list = stamp_list.replace("-", "x")
    stamp_list = stamp_list.replace(":", "x")
    stamp_list = stamp_list.replace(" ", "x")
    stamp_list = stamp_list.replace(".", "x")
    stamp_list = stamp_list.split("x")

    # Loop over the elements of the datetime string and concatenate them with the corresponding symbols
    stamp_str = ""
    for i in range(len(stamp_list)):
        if bool_list[i]:
            stamp_str += stamp_list[i]
            if i < len(stamp_list) and sum(bool_list[i+1:]) > 0:
                stamp_str += separator_list[i]

    return stamp_str


__start_stack = []
__start_named = {}


def tic(name: str = None):
    if name is None:
        __start_stack.append(time())
    else:
        __start_named[name] = time()


def toc(text: str = None, decimals: int = 6, verbose=None) -> float:
    if text is None:
        start = __start_stack.pop()
    else:
        try:
            start = __start_named.pop(text)
        except KeyError:
            start = __start_stack.pop()

    elapsed = time() - start

    if text is None:
        text = "Elapsed time"
    elif text == "":
        return elapsed

    print2(f"{text}: {elapsed:.{decimals + 1}}s", verbose=verbose)
    return elapsed


class tictoc:
    def __init__(self, text: str = None, decimals: int = 6, verbose=None):
        self.verbose = verbose_level_wrapper(verbose)
        self.text = "" if text is None else text
        self.decimals = decimals

    def __enter__(self):
        tic()
        print2(self.text+"...", verbose=self.verbose)

    def __exit__(self, *args):
        toc(text=None, decimals=self.decimals, verbose=self.verbose.add_level(1))
