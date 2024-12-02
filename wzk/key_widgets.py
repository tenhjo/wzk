import time

import numpy as np

try:
    from pynput import keyboard  # TODO: remove this after linux install works fine
except ImportError:
    keyboard = None


from wzk import np2, printing, ltd


class KeyListener:
    def __init__(self, key2callback: dict = None, combine_to_words=True, pass_key=False, start_listening=True):
        self.key2callback = key2callback

        self.k = ""
        self._dt = 1  # seconds, threshold after which a word stops
        self._time_of_last_press = -1
        self.combine_to_words = combine_to_words
        self.pass_key = pass_key

        self.listener = None
        self.is_listening = False

        if start_listening:
            self.start_listening()

    @staticmethod
    def wrapper_key(key):
        try:
            k = key.char  # single-char keys
        except AttributeError:
            k = key.name  # other keys
        return k

    def add_callback(self, key2callback: dict = None, start_listening=True):
        self.key2callback.update(key2callback)
        if start_listening:
            self.start_listening()

    def start_listening(self, on_press=True, on_release=False):
        if self.is_listening:
            self.stop_listening()

        if on_press and not on_release:
            self.listener = keyboard.Listener(on_press=self.on_press)
        elif on_release and not on_press:
            self.listener = keyboard.Listener(on_release=self.on_press)
        else:

            raise NotImplementedError

        self.listener.start()
        self.is_listening = True

    def stop_listening(self):
        self.listener.stop()
        self.listener.join()
        self.is_listening = False

    def in_keys(self, k):
        return [k in keys[:len(k)] for keys in self.key2callback]

    def __combine_to_words(self, k):
        if not self.combine_to_words:
            self.k = k
            return

        dt = time.time() - self._time_of_last_press
        self._time_of_last_press = time.time()

        # reset if dt is too large
        if dt < self._dt:
            self.k += k

        else:
            self.k = k
            return

        # reset if string does not match any keys
        if sum(self.in_keys(self.k)) == 0:
            self.k = k
            return

        # reset if the string is too long
        max_key_len = max([len(str(k)) for k in self.key2callback])
        if len(self.k) > max_key_len:
            self.k = k
            return

    def on_press(self, k):
        k = self.wrapper_key(k)

        self.__combine_to_words(k)
        if self.k in self.key2callback:
            if self.pass_key:
                self.key2callback[self.k](self.k)
            else:
                self.key2callback[self.k]()

            if sum(self.in_keys(self.k)) == 1:
                self.k = ""


class KeySlider(KeyListener):
    def __init__(self, callback, step, mi, ma, x=None, periodic=False,
                 keys=None, start_listening=True, dtype=float, verbose=0, _factor=20):
        self._factor = _factor
        if x is None:
            self.value = (ma - mi) / 2
        else:
            self.value = x

        self.verbose = verbose
        self.callback = callback
        self.step = step
        self.min = mi
        self.max = ma

        self.periodic = periodic
        self.dtype = dtype

        self.factor = self._update_factor()

        self.keys = keys
        super().__init__(key2callback={}, combine_to_words=False, start_listening=False, pass_key=True)
        self.add_callback(key2callback=self.get_key2callback(), start_listening=start_listening)

    def get_key2callback(self):
        if self.keys is None:
            k2k = dict(left="left", right="right", up="up", down="down")
        else:
            k2k = self.keys
        k2k_i = ltd.invert_dict(k2k)
        k2s = dict(left=(-1, 0), right=(+1, 0), up=(-1, 1), down=(+1, 1))

        key2callback = {k2k[k]: lambda kk: self.update(step=k2s[k2k_i[kk]]) for k in k2k}
        return key2callback

    def update(self, step):
        self.step_value(step=step)
        self.clip_value()
        self.cast_value()

        if self.callback is not None:
            if self.verbose > 0:
                print(self.value)
            self.callback(self.value)

    def step_value(self, step):
        step = self.step * step[0] * (1 + ((self.factor - 1) * step[1]))
        self.value = self.value + step

    def clip_value(self):
        if self.periodic:
            self.value = np2.clip_periodic(x=self.value, a_min=self.min, a_max=self.max)
        else:
            self.value = np.clip(a=self.value, a_min=self.min, a_max=self.max)

    def cast_value(self):
        if self.dtype == float:
            self.value = float(self.value)
        elif self.dtype == int:
            self.value = int(self.value)

    def set_value(self, v):
        self.value = v

    def _update_factor(self):
        self.factor = int((self.max - self.min) / self._factor)
        return self.factor

    def set_limits(self, mi=None, ma=None):
        if mi is not None:
            self.min = mi
        if ma is not None:
            self.max = ma
        self._update_factor()


class BoxLimitsKeySlider:

    def __init__(self, limits, x=None, names=None, callback_x=None, callback_j=None):
        # Parameters
        self.__m = 100

        # Args
        self.x = x
        if self.x is None:
            self.x = limits[:, 0] + (limits[:, 1] - limits[:, 0]) / 2
        self.n = len(self.x)
        self.j = 0
        self.limits = limits
        self.names = names
        self.callback_x = callback_x
        self.callback_j = callback_j

        self.slider_mode = "x"

        # Create KeyListeners
        key2callback = {str(i): lambda k: self.change_j(k) for i in range(self.n)}
        key2callback["p"] = lambda k: self.return_x()
        key2callback["j"] = lambda k: self.change_slider_mode("j")
        key2callback["x"] = lambda k: self.change_slider_mode("x")

        self.ks_x = KeySlider(callback=self.change_xj, step=1, mi=0, ma=self.__m, periodic=True, start_listening=False)
        self.ks_x.add_callback(key2callback=key2callback, start_listening=False)
        self.ks_x.pass_key = True
        self.ks_x.combine_to_words = True

        ks_j = KeySlider(callback=self.change_j, step=1, mi=0, ma=self.n, periodic=True, x=0,
                         keys=dict(left="w", right="s"), start_listening=False)
        self.ks_x.add_callback(key2callback=ks_j.key2callback, start_listening=True)

        self.print_state(clear_previous=False)
        self.change_j(j=0)

    def stop_listening(self):
        self.ks_x.stop_listening()

    def return_x(self, verbose=1):
        if verbose > 0:
            print(repr(self.x), end="")
        return self.x

    def get_x_discrete(self):
        x = (self.x[self.j] - self.limits[self.j, 0]) / (self.limits[self.j, 1] - self.limits[self.j, 0])
        x = x * self.__m
        return x

    def change_j(self, j):
        self.j = int(j)
        self.ks_x.set_value(self.get_x_discrete())

        self.print_state()
        self._callback("j")

    def change_xj(self, x):
        if self.slider_mode == "j":
            self.j = int(x)

        elif self.slider_mode == "x":
            dx = (self.limits[self.j, 1] - self.limits[self.j, 0]) / self.__m
            self.x[self.j] = self.limits[self.j, 0] + x * dx

        else:
            raise ValueError

        self._callback(self.slider_mode)
        self.print_state()

    def change_slider_mode(self, mode):

        self.slider_mode = mode
        if mode == "j":
            self.ks_x.set_limits(mi=0, ma=self.n)
            self.ks_x.factor = 1
            self.ks_x.set_value(self.j)

        elif mode == "x":
            self.ks_x.set_limits(mi=0, ma=self.__m)
            self.ks_x.set_value(self.get_x_discrete())

        else:
            raise ValueError

    def _callback(self, mode):
        if mode == "j":
            if self.callback_j is not None:
                self.callback_j(self.x, self.j)
        elif mode == "x":
            if self.callback_x is not None:
                self.callback_x(self.x, self.j)

        else:
            raise ValueError

    def print_state(self, clear_previous=True):
        if clear_previous:
            printing.clear_previous_lines(n=self.n + 1)

        s = printing.x_and_limits2txt(x=self.x, limits=self.limits, names=self.names)
        s = s.split("\n")
        s[self.j] += f"  <- {self.j}"
        s = "\n".join(s)
        print(s)


def __get_samples_x_limits(n):
    x = np.random.random(n)
    limits = np.zeros((n, 2))
    limits[:, 0] = x - np.random.random(n)
    limits[:, 1] = x + np.random.random(n)
    return x, limits


def try_x_and_limits2txt():
    x, limits = __get_samples_x_limits(n=4)
    s = printing.x_and_limits2txt(x=x, limits=x.limits)
    print(s)


def try_BoxLimitsKeySlider():
    print("test BoxLimitsKeySlider with Arrow Keys and `x j p`")

    names = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5"]
    x, limits = __get_samples_x_limits(n=5)
    kl = BoxLimitsKeySlider(limits=limits, x=x, names=names)
    input("Press enter to to next test")
    kl.stop_listening()


def try_KeySlider():
    print("test KeySlider with Arrow Keys")

    def fun(i):
        print(f"fun({i})")

    kl = KeySlider(callback=fun, step=1, mi=0, ma=100, periodic=True)
    input("Press enter to to next test")
    kl.stop_listening()


def try_KeyListener():
    print("test KeyListener with `w,a,s,d`")
    key2callback = dict(w=lambda: print("ww"),
                        a=lambda: print("aa"),
                        s=lambda: print("ss"),
                        d=lambda: print("dd")
                        )
    kl = KeyListener(key2callback=key2callback)
    input("Press enter to to next test")
    kl.stop_listening()


if __name__ == "__main__":
    pass

    try_KeyListener()

    try_KeySlider()

    try_BoxLimitsKeySlider()

    input()
