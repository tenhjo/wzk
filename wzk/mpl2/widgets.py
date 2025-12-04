from matplotlib import widgets
from wzk import math2


# Widgets
def create_button(fig, axes, listener_fun, name="button"):
    b_ax = fig.axes(axes)
    b = widgets.Button(b_ax, name)
    b.on_clicked(listener_fun)
    return b


def create_key_slider(*, ax, callback,
                      label="", valfmt=None, valmin=0, valmax=10, valinit=0, valstep=1,
                      fast_step=10):

    slider = widgets.Slider(ax=ax,
                            label=label, valfmt=valfmt,
                            valmin=valmin, valmax=valmax, valinit=valinit, valstep=valstep)

    def cb_key(event):
        val = slider.val
        if event.key == "right":
            slider.set_val(math2.modulo(val+valstep, low=valmin, high=valmax+1))

        if event.key == "left":
            slider.set_val(math2.modulo(val-valstep, low=valmin, high=valmax+1))

        if event.key == "up":
            slider.set_val(math2.modulo(val+fast_step*valstep, low=valmin, high=valmax+1))

        if event.key == "down":
            slider.set_val(math2.modulo(val-fast_step*valstep, low=valmin, high=valmax+1))

    slider.on_changed(callback)
    keyboard = ax.get_figure().canvas.mpl_connect("key_press_event", cb_key)
    return slider, keyboard
