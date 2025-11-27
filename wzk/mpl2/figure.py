from typing import Literal

import os
import shutil
from typing import Union, Optional

import numpy as np

from wzk.mpl2.backend import plt
from wzk.mpl2.axes import set_ax_limits
from wzk.mpl2.move_figure import move_fig

from wzk import files, ltd, math2, printing, strings

import matplotlib as mpl
from matplotlib import axes  # noqa
from matplotlib import figure  # noqa

shape_1c_ieee = [3 + 1 / 2, (3 + 1 / 2) / math2.GOLDEN_RATIO]
shape_2c_ieee = [7 + 1 / 16, (7 + 1 / 16) / math2.GOLDEN_RATIO]

axes_type = mpl.axes.Axes


def ax_wrapper(ax: Union[dict, mpl.axes.Axes]):
    if ax is None:
        return new_fig()[1]

    elif isinstance(ax, dict):
        return new_fig(**ax)[1]

    elif isinstance(ax, mpl.axes.Axes):
        return ax

    else:
        raise ValueError


def figsize_wrapper(width, height=None, height_ratio=1/math2.GOLDEN_RATIO):
    # https://www.ieee.org/content/dam/ieee-org/ieee/web/org/pubs/eic-guide.pdf
    if isinstance(width, str):
        if width.lower() == "ieee1c":
            width = shape_1c_ieee[0]
        elif width.lower() == "ieee2c":
            width = shape_2c_ieee[0]
        else:
            raise ValueError

    elif isinstance(width, (float, int)):
        pass
    else:
        raise ValueError

    height = width * height_ratio if height is None else height

    return width, height


def new_fig(width=shape_2c_ieee[0], height=None, h_ratio=1 / math2.GOLDEN_RATIO,
            n_rows: int = 1, n_cols: int = 1,
            share_x="none", share_y="none",  # : bool or {'none', 'all', 'row', 'col'},
            aspect: Literal["auto", "equal"] = "auto",
            limits=None,
            title=None,
            position=None, monitor=-1,
            kwargs_subplots=None,
            **kwargs):

    fig = plt.figure(figsize=figsize_wrapper(width=width, height=height, height_ratio=h_ratio), **kwargs)

    if kwargs_subplots is None:
        kwargs_subplots = {}

    ax = fig.subplots(nrows=n_rows, ncols=n_cols, sharex=share_x, sharey=share_y, **kwargs_subplots)  # noqa

    if isinstance(ax, np.ndarray):
        for i in np.ndindex(*np.shape(ax)):
            ax_i = ax[i]
            assert isinstance(ax_i, mpl.axes.Axes)
            ax_i.set_aspect(aspect)
            set_ax_limits(ax=ax, limits=limits)

    else:
        ax.set_aspect(aspect)
        set_ax_limits(ax=ax, limits=limits)

    if title is not None:
        fig.suptitle(title)

    move_fig(fig=fig, position=position, monitor=monitor)
    return fig, ax


def save_fig(file: str = None, fig: mpl.figure.Figure = None, formats: Union[str, tuple] = None,
             dpi: int = 600, bbox: Optional[str] = "tight", pad: float = 0.1,
             save: bool = True, replace: bool = True, show: bool = False, copy2cb: bool = False,
             verbose: int = 1, **kwargs: object) -> object:
    """
    adaption of the matplotlib 'savefig' function with some added convenience.
    bbox = tight / standard (standard does not crop but saves the whole figure)
    pad: padding applied to the thigh bounding box in inches
    """

    if not save:
        return

    if fig is None:
        fig = plt.gcf()

    if file is None:
        file = get_fig_suptitle(fig=fig)

    dir_name = os.path.dirname(file)
    if dir_name != "":
        files.mkdirs(directory=dir_name)

    file, ext = os.path.splitext(file)
    if ext == "":
        ext = tuple()
    else:
        ext = tuple([ext[1:]])

    if formats is None:
        formats = tuple()
    formats = ltd.atleast_tuple(formats, convert=False)
    formats = set(formats)
    formats = formats.union(set(ext))

    formats = list(formats)
    for f in formats:
        file_f = f"{file}.{f}"

        if replace or not os.path.isfile(path=file_f):
            fig.savefig(file_f, format=f, bbox_inches=bbox, pad_inches=pad,  dpi=dpi, **kwargs)
            if verbose >= 1:
                print(f"{file_f} saved")
        else:
            print(f"{file_f} already exists")

    if show:
        files.start_open(file=f"{file}.{formats[0]}")

    if copy2cb:
        files.copy2clipboard(file=f"{file}.{formats[0]}")


def save_ani(file: str, fig: mpl.figure.Figure, ani,
             n: int, fps: int = 30, dpi: int = 300, bbox: str = None,):
    dir_temp = os.path.split(file)[0]
    if dir_temp == "":
        dir_temp = os.getcwd()
        file = dir_temp + "/" + file
    dir_temp += "/" + strings.uuid4()

    if isinstance(n, int):
        n = np.arange(n)

    for nn in n:
        printing.progress_bar(i=nn, n=n[-1] + 1)
        ani(nn)
        save_fig(file="{}/frame{:0>6}".format(dir_temp, nn), fig=fig, formats=("png", ), dpi=dpi, bbox=bbox,
                 verbose=0)

    os.system(f"ffmpeg -r {fps} -i {dir_temp}/"
              f'frame%06d.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white" {file}.mp4')
    shutil.rmtree(dir_temp)


def save_all(directory: str = None, close: bool = False, **kwargs):
    if directory is None:
        directory = input("Directory to which the figures should be saved:")

    fig_nums = plt.get_fignums()
    for n in fig_nums:
        fig = plt.figure(num=n)
        title = get_fig_suptitle(fig=fig)
        title = "" if title is None else title
        save_fig(file=f"{directory}N{n}_{title}", fig=fig, **kwargs)

    if close:
        close_all()


# noinspection PyProtectedMember
def get_fig_suptitle(fig: mpl.figure.Figure):
    try:
        return fig._suptitle._text  # type: ignore
    except AttributeError:
        return ""


def close_all():
    plt.close("all")


def subplot_grid(n: int, squeeze: bool = False, **kwargs):
    n_rows, n_cols = math2.get_mean_divisor_pair(n)

    if n >= 7 and n_rows == 1:
        n_rows, n_cols = math2.get_mean_divisor_pair(n+1)

    _, ax = new_fig(n_rows=n_rows, n_cols=n_cols, **kwargs)

    if not squeeze:
        ax = np.atleast_2d(ax)
    return ax


def test_pdf2latex():
    fig, ax = new_fig(scale=0.5)
    ax.plot(np.random.random(20))
    ax.set_xlabel("Magnetization")
    save_fig(file="/Users/jote/Documents/Vorlagen/LaTeX Vorlagen/IEEE-open-journal-template/aaa", fig=fig,
             formats=("pdf",))
