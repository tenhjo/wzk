import numpy as np
import pandas as pd

from wzk.mpl2 import (set_borders, change_tick_appearance, elongate_ticks_and_labels,
                      CurlyBrace, make_every_box_fancy)

from wzk import ltd

from matplotlib import ticker


__style = dict(newline_size=1,
               perc_in_label=False, abs_in_xt=False,
               newline_in_legend=" ", newline_in_label=" ",
               absolut_right=True,
               fontweight_label=1)


def add_absolute_ax(ax, a, f=1, position="right"):
    def perc2abs(x):
        return x * a / 100 * f

    def abs2perc(x):
        return x / a * 100 / f

    if position == "right":
        ax.secondary_yaxis("right", functions=(perc2abs, abs2perc))
    elif position == "top":
        ax.secondary_xaxis("top", functions=(perc2abs, abs2perc))
    else:
        raise NotImplementedError


def value_counts(s, ordering=None):
    uc = s.value_counts()
    v = uc.index.values
    c = uc.values

    if ordering is not None:
        if isinstance(ordering[0], str):
            ordering = ltd.get_indices(li=v, el=ordering)
        v = v[ordering]
        c = c[ordering]

    return v, c


def handle_newline(newline, n):
    newline2 = [False] * n
    if newline is None:
        pass
    elif isinstance(newline, int):
        newline2[newline] = True
    elif isinstance(newline, list):
        for i in newline:
            newline2[i] = True
        return newline2

    return newline2


def mosaic_plot(ax, df, col_x, col_y,
                orderings_x=None, orderings_y=None, newline=None, ignore_na=True,
                style=None):

    # Handle Arguments
    df_x = df[col_x].values
    df_y = df[col_y].values

    if ignore_na:
        idx_not_na = ~np.logical_or(pd.isna(df_x), pd.isna(df_y))
        df_x = df_x[idx_not_na]
        df_y = df_y[idx_not_na]
    else:
        idx_not_na = np.ones_like(df_x, dtype=bool)

    val_x, counts_x = value_counts(df[col_x][idx_not_na], ordering=orderings_x)
    val_y, counts_y = value_counts(df[col_y][idx_not_na], ordering=orderings_y)

    n_x = len(val_x)
    n_y = len(val_y)

    newline = handle_newline(newline=newline, n=n_x)

    if style["colors"] is None:
        colors = [None] * n_y
    else:
        colors = style["colors"]

    gap = 0.0
    perc_x = counts_x / counts_x.sum() * 100
    width = perc_x - gap
    x_idx = perc_x / 2 + gap / 2
    x_idx[1:] += np.cumsum(perc_x)[:-1]

    bottom = np.zeros(n_x)
    temp = np.zeros(n_x)
    bool_x = np.array([df_x == vx for vx in val_x])

    for i, vy in enumerate(val_y):
        bool_y = np.array(df_y == vy)

        for j, bx in enumerate(bool_x):
            temp[j] = np.logical_and(bx, bool_y).sum() / np.sum(bx) * 100

        vy = vy.replace("\n", style["newline_in_legend"])
        vy = vy.replace("u.", "undecided")

        ax.bar(x_idx, temp, width=width, bottom=bottom,
               label=vy + " ({:.1f}%)".format(bool_y.mean()*100) if style["perc_in_label"] else vy,
               color=colors[i])

        curly_hl = style["curly_highlight"]
        if curly_hl is not None and i in curly_hl[:, 0]:
            jj = curly_hl[curly_hl[:, 0] == i, 1]
            for j in jj:

                cb = CurlyBrace(p=(x_idx[j] - width[j]/3, bottom[j]),
                                x0=(x_idx[j] - width[j]/3, bottom[j]+temp[j]),
                                x1=(x_idx[j] - width[j]/3 + 1, bottom[j]+temp[j]/2),
                                zorder=10)
                ax.add_patch(cb)
                ax.annotate(xy=(x_idx[j] - width[j]/3+1, bottom[j] + temp[j]/2), s="  {:.1f}%".format(temp[j]),
                            va="center", ha="left")

        bottom += temp

    # Axis Labels
    # x
    labels_xb = [lxb.replace("\n", style["newline_in_label"]) for lxb in val_x.tolist()]
    labels_xb = [lxb.replace(".u", "undecided") for lxb in labels_xb]

    labels_xt = ["{} ({:.1f}%)".format(c, c_perc) if style["abs_in_xt"] else "{:.1f}%".format(c_perc)
                 for c, c_perc in zip(counts_x, counts_x/sum(counts_x)*100)]
    for i, (lb, lt) in enumerate(zip(labels_xb, labels_xt)):
        if newline[i]:
            labels_xb[i] = f"\n{lb}"
            labels_xt[i] = f"{lt}\n"

    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels_xb)
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks(x_idx)
    ax_top.set_xticklabels(labels_xt)
    ax.set_xlim(0, 100.5)
    for i in np.nonzero(newline)[0]:
        change_tick_appearance(ax=ax, position="bottom", v=i, size=style["newline_size"])
        change_tick_appearance(ax=ax_top, position="top", v=i, size=style["newline_size"])

    # y
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(10, 91, 20), minor=True)
    ax.set_yticklabels([str(n) + "%" for n in np.arange(0, 101, 20)])
    ax.set_ylim(0, 100.5)

    if style["absolut_right"]:
        add_absolute_ax(ax=ax, a=sum(counts_y))

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower left")
    ax.set_xlabel(col_x, fontweight=style["fontweight_label"])
    ax.set_ylabel(col_y, fontweight=style["fontweight_label"])

    make_every_box_fancy(ax=ax)
    return ax


def bar_plot(ax, df, col, ordering=None, x_label="", newline=None,
             style=None):

    labels, counts = value_counts(df[col], ordering=ordering)
    bar_height = counts / counts.sum()

    if style["orientation"] == "vertical":

        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
        ax.bar(np.arange(len(labels)), bar_height, color=style["colors"])
        ax.set_xticks(np.arange(len(labels)))
        elongate_ticks_and_labels(ax=ax, labels=labels, newline=newline)

        add_absolute_ax(ax=ax, a=counts.sum(), f=100, position="right")
        ax.set_xlabel(col + x_label, fontweight=style["fontweight_label"])

    elif style["orientation"] == "horizontal":
        set_borders(left=0.2, right=0.87, top=0.9, bottom=0.1)

        ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
        ax.barh(np.arange(len(labels)), bar_height, color=style["colors"])
        ax.set_yticks(np.arange(len(labels)))
        elongate_ticks_and_labels(ax=ax, labels=labels, newline=newline, axis="y")

        ax.set_ylabel(col + x_label, rotation=-90, fontweight=style["fontweight_label"])
        add_absolute_ax(ax=ax, a=counts.sum(), f=100, position="top")

    else:
        raise ValueError

    return ax


def multi_bar_plot(ax, df, cols, ordering=None, colors=None, newline=None):

    bar_width = 0.8
    labels, counts = ltd.change_tuple_order((value_counts(df[col], ordering=ordering) for col in cols))
    labels = labels[0]
    c_max = max([c.sum() for c in counts])
    heights = [c / c_max for c in counts]

    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))

    w = g = -1
    for i, (h, c) in enumerate(zip(heights, colors)):
        g = 0.05
        w = (bar_width - (len(cols) - 1) * g) / len(cols)
        ax.bar(np.arange(len(labels))+i*(w+g), h, color=c, width=w, label=cols[i])

    ax.set_xticks(np.arange(len(labels)) + (len(cols)-1) * (w + g) / 2)
    elongate_ticks_and_labels(ax=ax, labels=labels, newline=newline)

    add_absolute_ax(ax=ax, a=c_max, f=100, position="right")
    ax.legend()
    
    return ax


def multi_bar_overlay_plot(ax, df, cols, ordering=None, colors=None, newline=None,
                           shift=0.05, sort=True):
    bar_width = 0.8

    colors = np.array(colors)
    labels, counts = ltd.change_tuple_order((value_counts(df[col], ordering=ordering) for col in cols))
    labels = labels[0]
    c_max = max([c.sum() for c in counts])
    heights = np.array([c / c_max for c in counts])

    if isinstance(sort, bool) and sort:
        sort_matrix = np.array([np.argsort(h) for h in heights.T]).T
    else:
        if isinstance(sort, bool):
            sort = np.arange(len(cols))
        sort_matrix = np.array(sort)[:, np.newaxis].repeat(len(labels), axis=1)

    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))

    w = bar_width - len(cols)*shift
    for j, i in enumerate(reversed(sort_matrix)):
        ax.bar(np.arange(len(labels))+j*shift, heights[i, range(len(labels))], color=colors[i], label=cols[i[0]],
               width=w)

    ax.set_xticks(np.arange(len(labels)) + ((len(cols)-1) * shift) / 2)
    elongate_ticks_and_labels(ax=ax, labels=labels, newline=newline)

    add_absolute_ax(ax=ax, a=c_max, f=100, position="right")
    ax.legend()
    
    return ax


def multiple_choice_bar(ax, df, cols, ordering=None, colors=None, x_label="", newline=None):
    labels, counts = ltd.change_tuple_order((value_counts(df[col], ordering=ordering) for col in cols))

    labels = np.array([la[0] for la in labels])
    counts = np.array([co[0] for co in counts])

    b_all = np.ones(len(df), dtype=bool)
    for c in cols:
        b_all = np.logical_or(b_all, pd.isna(df[c]).values)
    heights = counts / b_all.sum()

    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))

    ax.bar(np.arange(len(labels)), heights, color=colors)

    ax.set_xticks(np.arange(len(labels)))
    elongate_ticks_and_labels(ax=ax, labels=labels, newline=newline)

    add_absolute_ax(ax=ax, a=b_all.sum(), f=100, position="right")
    ax.set_xlabel(x_label)

    return ax


def conditional_bar_plot(ax, df, col, col_cond, val_cond, ordering=None, newline=None, style=None):
    bool_cond = pd.DataFrame(df[col_cond] == val_cond).values
    val_cond = val_cond.replace("\n", ", ")
    return bar_plot(ax=ax, df=df.iloc[bool_cond, :], col=col, 
                    newline=newline, ordering=ordering,
                    style=style,
                    x_label=f"\n(given '{col_cond}'='{val_cond}')")


def pie_plot(ax, df, col, ordering=None, newline=None, style=None):
    labels, counts = value_counts(df[col], ordering=ordering)

    patches, hl_o, hl_i = ax.pie(counts, colors=style["colors"], startangle=90, counterclock=False,
                                 autopct=style["autopct"],
                                 rotatelabels=False, pctdistance=0.6)

    newline = handle_newline(newline=newline, n=len(labels))

    for i, hl in enumerate(hl_i):
        t = hl.get_text()
        hl.set_text(f"{counts[i]} ({t})\n{labels[i]}")

        if newline[i]:
            xy = np.array(hl.get_position())
            xy = xy + xy * 1/2
            hl.set_position(xy)
    
    return ax
