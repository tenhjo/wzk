import numpy as np
import matplotlib.patches as patches
from typing import Callable

from wzk import np2
from wzk.mpl2.axes import get_aspect_ratio
from wzk.mpl2.geometry import plot_coordinate_frames
from wzk.mpl2.figure import plt

from wzk.spatial.transform_2d import v2dcm


class DummyPatch:
    __slots__ = ("figure",
                 "axes",
                 "set_animated",
                 "contains",
                 "set_visible",
                 "get_visible")


class DraggablePatch(DummyPatch):
    lock = None  # only one can be animated at a time

    def __init__(self, ax, vary_xy=(True, True), limits=None, callback=None,
                 wsad=None, **kwargs):
        super().__init__()
        ax.add_patch(self)

        self.vary_xy = np.array(vary_xy)
        # Patches already have an attribute callback, add_callback()
        self.callback_drag = [callback] if isinstance(callback, Callable) else callback
        self.limits = limits

        self.wsad = wsad
        self.__wsad_step = 0.1
        self.press = None
        self.background = None

        # Connections
        self.cid_press = None
        self.cid_release = None
        self.cid_motion = None
        self.cid_key = None
        self.connect()

    def set_callback_drag(self, callback):
        if isinstance(callback, Callable):
            self.callback_drag = [callback]
        elif isinstance(callback, list):
            self.callback_drag = callback

    def add_callback_drag(self, callback):
        if self.callback_drag is None:
            self.set_callback_drag(callback=callback)
        else:
            self.callback_drag.append(callback)

    def get_callback_drag(self):
        return self.callback_drag

    def get_xy_drag(self):
        raise NotImplementedError

    def set_xy_drag(self, xy):
        raise NotImplementedError

    def apply_limits(self, xy):
        if self.limits is not None:
            return np.clip(xy, a_min=self.limits[:, 0], a_max=self.limits[:, 1])
        else:
            return xy

    def set_limits(self, limits=None):
        self.limits = limits
        self.set_xy_drag(xy=self.apply_limits(self.get_xy_drag()))

    def connect(self):
        self.cid_press = self.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = self.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        if self.wsad:
            self.cid_key = self.figure.canvas.mpl_connect("key_press_event", self.on_key)

    def disconnect(self):
        self.figure.canvas.mpl_disconnect(self.cid_press)
        self.figure.canvas.mpl_disconnect(self.cid_release)
        self.figure.canvas.mpl_disconnect(self.cid_motion)
        if self.wsad:
            self.figure.canvas.mpl_disconnect(self.cid_key)

    def on_press(self, event):
        if event.inaxes != self.axes:
            return
        if DraggablePatch.lock is not None:
            return
        contains, attrd = self.contains(event)
        if not contains:
            return

        self.press = np.array(self.get_xy_drag()), np.array([event.xdata, event.ydata])
        DraggablePatch.lock = self

        # Draw everything but the selected rectangle and store the pixel buffer
        canvas = self.figure.canvas
        axes = self.axes
        self.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.axes.bbox)

        # Now redraw just the rectangle
        axes.draw_artist(self)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        if DraggablePatch.lock is not self:
            return
        if event.inaxes != self.axes:
            return

        xy_patch, xy_press = self.press

        dxy = np.array([event.xdata, event.ydata]) - xy_press

        new_xy_patch = xy_patch + self.vary_xy * dxy

        self.set_xy_drag(self.apply_limits(xy=new_xy_patch))

        canvas = self.figure.canvas
        axes = self.axes

        # restore the background region
        if self.background is not None:
            canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):  # noqa
        """on release, we reset the press Measurements"""

        if DraggablePatch.lock is not self:
            return

        self.press = None
        DraggablePatch.lock = None

        # turn off the rect animation property and reset the background
        self.set_animated(False)
        self.background = None

        if self.callback_drag is not None:
            for cb_i in self.callback_drag:
                cb_i(self)

    def on_key(self, event):
        xy = self.get_xy_drag()
        if event.key == "left":
            xy = [xy[0] - self.__wsad_step, xy[1]]
        elif event.key == "right":
            xy = [xy[0] + self.__wsad_step, xy[1]]
        elif event.key == "up":
            xy = [xy[0], xy[1] + self.__wsad_step]
        elif event.key == "down":
            xy = [xy[0], xy[1] - self.__wsad_step]
        elif event.key == "a":
            self.__wsad_step *= 0.5
        elif event.key == "d":
            self.__wsad_step *= 2

        self.set_xy_drag(xy=self.apply_limits(xy=xy))
        if self.callback_drag is not None:
            for cb_i in self.callback_drag:
                cb_i(self)

        plt.show()

    def toggle_visibility(self, value=None):
        if value is None:
            self.set_visible(not self.get_visible())
        else:
            self.set_visible(bool(value))

        return self.get_visible()


class DraggableCircle(patches.Circle, DraggablePatch):
    def __init__(self,
                 ax,
                 xy: np.ndarray | list | tuple,
                 radius,
                 vary_xy=(True, True), callback=None,
                 limits=None,
                 wsad=None,
                 **kwargs):
        patches.Circle.__init__(self, xy=(xy[0], xy[1]), radius=radius, **kwargs)
        DraggablePatch.__init__(self, ax=ax, vary_xy=vary_xy, callback=callback, limits=limits, wsad=wsad)

    def get_xy_drag(self):
        return np.array(self.get_center()).flatten()

    def set_xy_drag(self, xy):
        self.set_center(xy=np.array(xy).flatten())  # noqa


class DraggableEllipse(patches.Ellipse, DraggablePatch):
    def __init__(self,
                 ax,
                 xy, width, height, angle=0,
                 vary_xy=(True, True), callback=None, limits=None,
                 **kwargs):
        """
        If fig_width_inch or height are None,
        they are computed to form a circle for the aspect and Measurements ratio of the axis.
        """
        if width is None or width == -1:
            width = get_aspect_ratio(ax) / ax.get_data_ratio() * height
        if height is None or height == -1:
            height = ax.get_data_ratio() / get_aspect_ratio(ax) * width

        patches.Ellipse.__init__(self, xy=xy, width=width, height=height, angle=angle, **kwargs)
        DraggablePatch.__init__(self, ax=ax, vary_xy=vary_xy, callback=callback, limits=limits)

    def get_xy_drag(self):
        return np.array(self.get_center()).flatten()

    def set_xy_drag(self, xy):
        # noinspection PyArgumentList
        self.set_center(xy=np.array(xy).flatten())


class DraggableRectangle(patches.Rectangle, DraggablePatch):
    def __init__(self, *,
                 ax,
                 xy, width, height, angle=0,
                 vary_xy=(True, True), callback=None, limits=None,
                 **kwargs):
        patches.Rectangle.__init__(self, xy=xy, width=width, height=height, angle=angle, **kwargs)
        DraggablePatch.__init__(self, ax=ax, vary_xy=vary_xy, callback=callback, limits=limits)

    def get_xy_drag(self):
        return np.array(self.get_xy()).flatten()

    def set_xy_drag(self, xy):
        # noinspection PyArgumentList
        self.set_xy(xy=np.array(xy).flatten())


class DraggablePatchList:
    def __init__(self):
        self.dp_list: list[DraggablePatch] = []

    def append(self, **kwargs):
        raise NotImplementedError

    def __getitem__(self, item):
        return self.dp_list[item]

    def __n_index_wrapper(self, i, n):
        if i is None:
            i = np.arange(n)
        elif isinstance(i, int):
            if i == -1:
                i = np.arange(len(self.dp_list))
            else:
                i = [i]

        n = int(np.max([n, np.size(i)]))

        return n, i

    @staticmethod
    def __value_wrapper(v, v_cur, n):
        if v is None:
            v = v_cur
        elif np.size(v) < n:
            v = np.full(n, v)

        if np.size(v) == 1:
            v = [v]

        return v

    def get_xy(self, idx=-1):
        n, idx = self.__n_index_wrapper(idx, len(self.dp_list))
        return np.vstack([dp.get_xy_drag() for i, dp in enumerate(self.dp_list) if i in idx])

    def get_callback(self, idx=-1):
        if idx is None or (isinstance(idx, int) and idx == -1):
            return np.vstack([dp.get_callback_drag for dp in self.dp_list])
        else:
            return np.vstack([dp.get_callback_drag for i, dp in enumerate(self.dp_list) if i in idx])

    def set_xy(self, xy=None, x=None, y=None, idx=None):

        if xy is not None:
            x, y = xy.T

        n = np2.max_size(x, y)
        n, idx = self.__n_index_wrapper(i=idx, n=n)

        xy_cur = self.get_xy(idx=idx)
        x = self.__value_wrapper(v=x, v_cur=xy_cur[:, 0], n=n)
        y = self.__value_wrapper(v=y, v_cur=xy_cur[:, 1], n=n)

        for ii, xx, yy in zip(idx, x, y):
            self.dp_list[ii].set_xy_drag(xy=(xx, yy))

    def __set_or_add_callback_drag(self, callback, idx, mode):
        n = np.size(callback)
        n, idx = self.__n_index_wrapper(i=idx, n=n)
        callback_cur = self.get_callback(idx=idx)
        callback = self.__value_wrapper(v=callback, v_cur=callback_cur, n=n)

        if mode == "set":
            for ii, cc in zip(idx, callback):
                self.dp_list[ii].set_callback_drag(callback=cc)
        elif mode == "add":
            for ii, cc in zip(idx, callback):
                self.dp_list[ii].add_callback_drag(cc)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def set_callback_drag(self, callback, idx=-1):
        self.__set_or_add_callback_drag(callback=callback, idx=idx, mode="set")

    def add_callback_drag(self, callback, idx=-1):
        self.__set_or_add_callback_drag(callback=callback, idx=idx, mode="add")

    def toggle_visibility(self, value=None):
        v = [dp.toggle_visibility(value=value) for dp in self.dp_list]
        return np2.unify(v)


class DraggableCircleList(DraggablePatchList):
    def __init__(self, ax, xy, radius, **kwargs):
        super().__init__()
        self.append(ax=ax, xy=xy, radius=radius, **kwargs)

    def append(self, ax, xy, radius, **kwargs):
        radius = np2.scalar2array(radius, shape=xy.shape[0])
        for xy_i, radius_i in zip(xy, radius):
            self.dp_list.append(DraggableCircle(ax=ax, xy=xy_i, radius=radius_i, **kwargs))


class DraggableEllipseList(DraggablePatchList):
    def __init__(self, ax,
                 xy, width, height, angle=0,
                 **kwargs):
        super().__init__()
        self.append(ax, xy=xy, width=width, height=height, angle=angle, **kwargs)

    def append(self, ax,
               xy, width, height, angle, **kwargs):
        width, height, angle = np2.scalar2array(width, height, angle, shape=xy.shape[0])
        for xy_i, width_i, height_i, angle_i in zip(xy, width, height, angle):
            self.dp_list.append(DraggableEllipse(ax=ax,
                                                 xy=xy_i, width=width_i, height=height_i, angle=angle_i, **kwargs))


class DraggableRectangleList(DraggablePatchList):
    def __init__(self, ax,
                 xy, width, height, angle=0,
                 **kwargs):
        super().__init__()
        self.append(ax, xy=xy, width=width, height=height, angle=angle, **kwargs)

    def append(self, ax,
               xy, width, height, angle, **kwargs):
        width, height, angle = np2.scalar2array(width, height, angle, shape=xy.shape[0])

        for xy_i, width_i, height_i, angle_i in zip(xy, width, height, angle):
            self.dp_list.append(DraggableRectangle(ax=ax,
                                                   xy=xy_i, width=width_i, height=height_i, angle=angle_i, **kwargs))


class DraggableFrame(DraggableCircleList):
    def __init__(self, ax, xy, scale, **kwargs):

        self.size = scale
        self.size_radius_center = 0.1
        self.size_radius_handle = 0.075
        self.size_position_handle = 0.8

        self.v = np.array([1, 0])
        xv = self.__x_v2xv(x=xy, v=self.v)

        super().__init__(ax, xy=np.vstack([xy, xv]), radius=np.array([self.size_radius_center * self.size,
                                                                      self.size_radius_handle * self.size]),
                         **kwargs)

        self.h = plot_coordinate_frames(ax=ax, x=xy, dcm=self.__v2dcm(self.v),
                                        mode="quiver", zorder=10, **kwargs)
        super().add_callback_drag(self.update_x, idx=0)
        super().add_callback_drag(self.update_v, idx=1)

    def update_x(self, *args):  # noqa
        xc, xv = self.get_xy()
        self.__update(xc=xc, v=self.v)

    def update_v(self, *args):  # noqa
        xc, xv = self.get_xy()
        self.v = xv - xc
        self.v = self.v / np.linalg.norm(self.v)
        self.__update(xc=xc, v=self.v)

    def __x_v2xv(self, x, v):
        return x + v * self.size * self.size_position_handle

    def __v2dcm(self, v):
        return v2dcm(v) * self.size

    def __update(self, xc, v):
        xv = self.__x_v2xv(x=xc, v=v)
        self.set_xy(xy=xv, idx=1)
        self.h = plot_coordinate_frames(x=xc, dcm=self.__v2dcm(v), h=self.h)

    def get_dcm(self):
        return v2dcm(self.v)

    def get_frame(self):
        f = np.eye(3)
        x = self.get_xy(idx=0)
        dcm = self.get_dcm()
        f[:2, :2] = dcm
        f[:2, -1] = x
        return f

    def add_callback_drag(self, callback, idx=-1):
        super().add_callback_drag(callback=lambda circle: callback(self), idx=idx)

    def set_callback_drag(self, callback, idx=-1):
        super().set_callback_drag(callback=lambda circle: callback(self), idx=idx)
