from wzk.mpl2 import new_fig


def plot_mean_var(mean, log_var, mean_limit=None, var_limit=None, **kwargs):

    if mean_limit is None:
        mean_limit = -6, 6

    if var_limit is None:
        var_limit = -10, 1

    n_random_var = mean.shape[-1]

    fig, axes_mean = new_fig(n_cols=1, n_rows=n_random_var, share_x="all")
    fig, axes_var = new_fig(n_cols=1, n_rows=n_random_var, share_x="all")
    if n_random_var == 1:
        axes_mean.hist(mean, **kwargs)
        axes_var.hist(log_var, **kwargs)
    else:
        for i in range(n_random_var):
            axes_mean[i].hist(mean[:, i], **kwargs)
            axes_var[i].hist(log_var[:, i], **kwargs)

        axes_mean[-1].set_xlim(mean_limit[0], mean_limit[1])
        axes_var[-1].set_xlim(var_limit[0], var_limit[1])

    return axes_mean, axes_var
