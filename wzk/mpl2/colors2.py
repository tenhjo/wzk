import numpy as np
from matplotlib import colors, cm


# Use :
# import cycler
# cycler('color', x)

# https://colorhunt.co/palette/15697
campfire = ["#311d3f", "#522546", "#88304e", "#e23e57"]

# https://colorhunt.co/palette/108152
sunset = ["#3a0088", "#930077", "#e61c5d", "#ffbd39"]

# https://colorhunt.co/palette/1504
blues4 = ["#48466d", "#3d84a8", "#46cdcf", "#abedd8"]

# https://learnui.design/tools/data-color-picker.html#single
blues10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


divergent_blue_red13 = np.array(["#395e8f" "#5a749f" "#798caf" "#97a4bf" "#b4bdcf" "#d2d6e0"
                                 "#f1f1f1"
                                 "#f3d9d1" "#f3c1b2" "#f1a994" "#ec9077" "#e6785a" "#de5e3e"])


divergent_br248842_13 = np.array(["#224488", "#4c5e99", "#6e79ab", "#8f95bc", "#afb3cd", "#d0d1df",
                                  "#f1f1f1",
                                  "#e2d2cb", "#d3b4a7", "#c29783", "#b07b62", "#9d5f41", "#884422"])

reds824_9 = np.array(["#882244",
                      "#983b56", "#a75168", "#b6677b", "#c57c8e", "#d492a2", "#e2a8b6", "#f1beca", "#ffd5df"])
reds842_9 = np.array(["#884422",
                      "#975634", "#a76747", "#b67a5b", "#c58c6f", "#d39f84", "#e2b299", "#f1c5af", "#ffd9c5"])

greens284_9 = np.array(["#228844",
                        "#3d9454", "#53a065", "#68ad75", "#7bb986", "#8fc597", "#a2d2a9", "#b6debb", "#c9ebcd"])
greens482_9 = np.array(["#448822",
                        "#579438", "#6aa04c", "#7cac60", "#8eb873", "#9fc487", "#b1d09c", "#c2ddb0", "#d4e9c5"])

blues248_9 = np.array(["#224488",
                       "#3e5596", "#5668a5", "#6c7ab4", "#828ec2", "#98a2d1", "#aeb6e0", "#c5cbf0", "#dbe0ff"])
blues428_9 = np.array(["#442288",
                       "#5b3896", "#704ea5", "#8564b4", "#9a7bc3", "#af93d2", "#c4abe1", "#d8c3f0", "#eddcff"])

rgb_248 = np.array(["#884422", "#228844", "#224488"])
rrggbb_248 = np.array(["#882244", "#884422", "#228844", "#448822", "#224488", "#442288"])

# palette_blues = ['#4864b0', '#6e7cb8', '#8f96bf', '#aeb1c6', '#cccccc']
mix6 = ["#b3c3e3", "#91a3c9", "#7084b0", "#506697", "#2f4a7e", "#002f66"]
mix5 = ["#0089b3", "#6f85d7", "#d66cc4", "#ff5d7a", "#ff8800"]


# TUM corporate design colors - http://portal.mytum.de/corporatedesign/index_print/vorlagen/index_farben
pallet_tum = {"blue_3": "#0065BD",  # Blue ordered from light to dark
              "blue_4":  "#005293",
              "blue_5":  "#003359",
              "blue_2":  "#64A0C8",
              "blue_1":  "#98C6EA",
              "grey_80": "#333333",
              "grey_50": "#808080",
              "grey_20": "#CCCCC6",
              "beige":   "#DAD7CB",
              "green":   "#A2AD00",
              "orange":  "#E37222"}

tum_blues5 = [pallet_tum["blue_5"],
              pallet_tum["blue_4"],
              pallet_tum["blue_3"],
              pallet_tum["blue_2"],
              pallet_tum["blue_1"]]

tum_mix5 = [pallet_tum["blue_3"],
            pallet_tum["orange"],
            pallet_tum["beige"],
            pallet_tum["green"],
            pallet_tum["grey_50"]]

# https://sashamaps.net/docs/resources/20-colors/
_20 = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
       "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9"]
_16 = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#42d4f4", "#f032e6", "#fabed4",
       "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3", "#000075", "#a9a9a9"]
_7 = ["#ffe119", "#4363d8", "#f58231", "#dcbeff", "#800000", "#000075", "#a9a9a9"]


def arr2rgba(img, cmap, vmin=None, vmax=None, mask=None, axis_order=None):
    img = __arr2rgba(arr=img, cmap=cmap, vmin=vmin, vmax=vmax)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape == img.shape[:2]:
            img[mask, 3] = 0
    if axis_order == "ij->yx":
        img = np.swapaxes(img, axis1=0, axis2=1)
    return img


def __arr2rgba(arr, cmap, vmin=None, vmax=None):
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    try:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)

    except (ValueError, AttributeError):
        cmap = colors.ListedColormap([cmap])
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)

    return sm.to_rgba(arr, bytes=True, norm=True)


def c_list_wrapper(c, n=20):
    c_list_dict = {"tum_blue443": [pallet_tum["blue_4"],
                                   pallet_tum["blue_4"],
                                   pallet_tum["blue_3"]],
                   "tum_blue4432": [pallet_tum["blue_4"],
                                    pallet_tum["blue_4"],
                                    pallet_tum["blue_3"],
                                    pallet_tum["blue_2"]],
                   "k_b_p_t_c_y": ["black", "blue", "xkcd:purple", "xkcd:teal", "cyan", "yellow"]}

    if c is None:
        c = "tum_blue443"

    if c in c_list_dict:
        c_list = c_list_dict[c]
    elif isinstance(c, str):
        c_list = [c]
    else:
        raise ValueError
    c_list *= n // len(c_list) + 1
    return c_list


def rgb2hex(rgb):
    h = (int(rgb[0] * 255) * 256 ** 2 +
         int(rgb[1] * 255) * 256 ** 1 +
         int(rgb[2] * 255) * 256 ** 0)
    return h
