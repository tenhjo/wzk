
from wzk.logger import log_print
import numpy as np
from matplotlib import transforms

from wzk.np2 import np_isinstance


def turn_ticklabels_off(ax, axes="xy"):
    """Remove the tick labels for all specified axes"""
    if "x" in axes:
        ax.axes.xaxis.set_ticklabels([])
    if "y" in axes:
        ax.axes.yaxis.set_ticklabels([])
    if "z" in axes:
        try:
            ax.axes.zaxis.set_ticklabels([])
        except AttributeError:
            pass


def turn_ticks_off(ax):
    turn_ticklabels_off(ax=ax)
    set_ticks_position(ax=ax, position="none")


def turn_ticks_and_labels_off(ax):
    turn_ticks_off(ax=ax)
    turn_ticklabels_off(ax=ax)


def set_spine_width(ax, lw):
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(lw)


def __position2tlbr(position="default"):
    """map keywords to booleans indicating which axis is active"""
    bottom = top = left = right = False
    if position == "all":
        bottom = top = left = right = True

    elif position == "default":
        bottom = left = True

    elif position == "inverse":
        top = right = True

    elif position == "none":
        pass

    else:
        if "bottom" in position:
            bottom = True
        if "top" in position:
            top = True
        if "left" in position:
            left = True
        if "right" in position:
            right = True

    return top, left, bottom, right


def __tlbr2positions(top, left, bottom, right):

    position_x = "none"
    if bottom and top:
        position_x = "both"
    elif bottom:
        position_x = "bottom"
    elif top:
        position_x = "top"

    position_y = "none"
    if left and right:
        position_y = "both"
    elif left:
        position_y = "left"
    elif right:
        position_y = "right"

    return position_x, position_y


def set_ticks_position(ax, position):
    """
    position: none, both,
           x: bottom, top
           y: left, right
    """
    position_x, position_y = __tlbr2positions(*__position2tlbr(position=position))

    ax.axes.xaxis.set_ticks_position(position=position_x)
    ax.axes.yaxis.set_ticks_position(position=position_y)

    # try:
    #     ax.axes.zaxis.set_ticks_position(position=position[2])
    # except AttributeError:
    #     pass


def set_labels_position(ax, position):

    top, left, bottom, right = __position2tlbr(position=position)
    ax.tick_params(labeltop=top, labelleft=left, labelbottom=bottom, labelright=right)


def get_ticks(ax, axis="x"):

    def ticks2arr(tt):
        return np.array([t for t in tt])

    if axis == "x":
        return ticks2arr(ax.get_xticks())
    elif axis == "y":
        return ticks2arr(ax.get_yticks())
    elif axis in ("xy", "both"):
        return ticks2arr(ax.get_xticks()), ticks2arr(ax.get_yticks())
    else:
        raise ValueError


def get_labels(ax, axis="x"):

    def labels2arr(labels):
        return np.array([la.get_text() for la in labels], dtype=object)

    if axis == "x":
        return labels2arr(ax.get_xticklabels())
    elif axis == "y":
        return labels2arr(ax.get_yticklabels())
    elif axis in ("xy", "both"):
        return labels2arr(ax.get_xticklabels()), labels2arr(ax.get_yticklabels())
    else:
        raise ValueError


def get_ticks_index(ax, axis, v, squeeze=True):
    assert axis == "x" or axis == "y"
    t = get_ticks(ax=ax, axis=axis)

    v = np.atleast_1d(v).copy()

    if np_isinstance(v, float):
        for i, vv in enumerate(v):
            temp = np.nonzero(np.isclose(t, vv))[0]
            v[i] = np.nan if np.size(temp) == 0 else temp[0]

    elif np_isinstance(v, int):
        pass

    else:
        raise ValueError

    v = np.array(v, dtype=int)
    if squeeze:
        v = np.squeeze(v)
    return v


def remove_ticks(ax, v, axis="x"):
    """
    Remove the ticks corresponding to the values in v from the axis.
    If the values are not found, they are ignored.
    """

    assert axis in ("x", "y", "xy", "both")

    def __remove_ticks(_axis):
        ticks = get_ticks(ax=ax, axis=_axis)
        i = get_ticks_index(ax=ax, axis=_axis, v=v, squeeze=False)
        i = i[i != np.nan]

        ticks2 = np.delete(ticks, i)
        return ticks2

    if "x" in axis or axis == "both":
        ax.set_xticks(__remove_ticks(_axis="x"))
    if "y" in axis or axis == "both":
        ax.set_yticks(__remove_ticks(_axis="y"))


def position2axis(position):
    p_list = ["bottom", "top", "left", "right"]
    ip = p_list.index(position)
    axis = "x" if ip//2 == 0 else "y"
    return axis, ip


def change_tick_appearance(ax, position, v, size=None, color=None):
    """
    x: bottom, top
    y: left, right
    """

    def __apply(h_i, s=None, c=None):
        if s is not None:
            h_i.set_markersize(s)
        if c is not None:
            h_i.set_markeredgecolor(c)

    # Handle different positions and axis
    axis, ip = position2axis(position=position)
    h = ax.xaxis.get_majorticklines() if axis == "x" else ax.yaxis.get_majorticklines()

    idx = get_ticks_index(ax=ax, axis=axis, v=v, squeeze=False)
    idx = np.array([np.ravel_multi_index((i, ip % 2), (len(h) // 2, 2)) for i in idx])

    for i in idx:
        __apply(h_i=h[i], s=size, c=color)


def change_label_appearance(ax, position, v, color, xt=0., yt=0.):

    def __apply(h_i, _color, _xt, _yt):
        if _color is not None:
            h_i.set_color(_color)
        __transform_label(ax=ax, h_label=h_i, xt=_xt, yt=_yt)
        # if c is not None:
        #     h_i.set_markeredgecolor(c)

    # Handle different positions and axis
    axis, ip = position2axis(position=position)
    h = ax.xaxis.get_ticklabels() if axis == "x" else ax.yaxis.get_ticklabels()

    idx = get_ticks_index(ax=ax, axis=axis, v=v, squeeze=False)

    for i in idx:
        __apply(h_i=h[i], _color=color, _xt=xt, _yt=yt)


def add_ticks(ax, ticks, labels=None, axis="x"):

    assert axis in ("x", "y", "xy", "both")
    if ticks is None or np.size(ticks) == 0:
        return

    def _as_list(v):
        if v is None:
            return []
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]

    def __add_ticks(_axis):
        ticks_old = get_ticks(ax=ax, axis=_axis)
        labels_old = get_labels(ax=ax, axis=_axis)

        _ticks = np.hstack((ticks_old, _as_list(ticks)))
        sort_idx = np.argsort(_ticks)
        _ticks = _ticks[sort_idx].tolist()

        if labels is not None and any(np.atleast_1d(labels)):
            # if len(labels_old) > len(ticks_old):  # two axes (bottom, top), (left, right)
            _labels = np.hstack((labels_old[:len(ticks_old)], _as_list(labels)))
            _labels = _labels[sort_idx].tolist()
        else:
            _labels = None

        set_ticks_and_labels(ax=ax, axis=_axis, ticks=_ticks, labels=_labels)

    if "x" in axis or axis == "both":
        __add_ticks(_axis="x")

    if "y" in axis or axis == "both":
        __add_ticks(_axis="y")


def set_ticks_and_labels(ax, ticks=None, labels=None, axis="x"):

    def __set_ticks(_set_ticks, _set_ticklabels, _ticks, _labels):

        if any(np.atleast_1d(_ticks)):
            _set_ticks(np.array(_ticks).tolist())

        if any(np.atleast_1d(_labels)):
            _set_ticklabels(np.array(_labels).tolist())

    assert axis in ("x", "y", "xy", "both")

    if "x" in axis or axis == "both":
        __set_ticks(_set_ticks=ax.set_xticks, _set_ticklabels=ax.set_xticklabels, _ticks=ticks, _labels=labels)

    if "y" in axis or axis == "both":
        __set_ticks(_set_ticks=ax.set_yticks, _set_ticklabels=ax.set_yticklabels, _ticks=ticks, _labels=labels)


def __transform_label(ax, h_label, xt=0., yt=0., ha=None, va=None):
    if xt != 0 or yt != 0:
        offset = transforms.ScaledTranslation(xt=xt, yt=yt, scale_trans=ax.get_figure().dpi_scale_trans)
        h_label.set_transform(h_label.get_transform() + offset)

    if va is not None:
        h_label.set_va(va)
    if ha is not None:
        h_label.set_ha(ha)


def transform_tick_labels(ax, xt=0., yt=0., rotation=0., axis="x", ha=None, va=None):

    if rotation != 0:
        ax.tick_params(axis=axis, rotation=rotation)

    if "x" in axis or "both" in axis:
        for label in ax.xaxis.get_majorticklabels():
            __transform_label(ax=ax, h_label=label, xt=xt, yt=yt, ha=ha, va=va)

    if "y" in axis or "both" in axis:
        for label in ax.yaxis.get_majorticklabels():
            __transform_label(ax=ax, h_label=label, xt=xt, yt=yt, ha=ha, va=va)


def handle_newline(newline, n):
    newline2 = [False] * n

    if newline is None:
        return None

    elif isinstance(newline, int):
        newline2[newline] = True
        return None

    elif isinstance(newline, (list, tuple, np.ndarray)):
        if len(newline) == n:
            return newline

        assert isinstance(newline[0], int)
        for i in newline:
            newline2[i] = True
        return newline2

    else:
        raise ValueError


def elongate_ticks_and_labels(ax, newline, labels=None, axis="x", position=None):
    newline_size, newline = newline
    assert axis == "x" or axis == "y", "axis must be 'x' or 'y'"
    if position is None:
        if axis == "x":
            position = "bottom"

        else:  # == 'y':
            position = "left"

    if labels is None:
        labels = get_labels(ax=ax, axis=axis)

    newline = handle_newline(newline=newline, n=len(labels))
    for i, lab in enumerate(labels):
        if newline[i]:
            nl = "\n" * newline[i]
            labels[i] = f"{nl}{lab}"
            change_tick_appearance(ax=ax, position=position, v=i, size=newline_size * newline[i])

    log_print(labels)

    if axis == "x":
        ax.set_xticks(ax.get_xticks())  # https://stackoverflow.com/a/68794383/7570817
        ax.set_xticklabels(labels)

    else:  # 'y'
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(labels, rotation=-90, ha="right", va="center")
