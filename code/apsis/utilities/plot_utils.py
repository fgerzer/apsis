__author__ = 'Frederik Diehl'
import matplotlib.pyplot as plt
plt.ioff()
import random
import os
from matplotlib.colors import colorConverter


def plot_lists(to_plot_list, fig_options=None, ax=None, plot_min=None,
               plot_max=None):
    """
    Plots several functions.

    Each entry of to_plot_list defines x, y, and format options.

    Parameters
    ----------
    to_plot_list : list of dicts
        Defines the functions to plot.
        Each entry must contain at least values for "x" and
        "y", and can contain values for "type", "label" and "color".
        x : list
            A list of x values
        y : list
            A list of y values
        type : string, optional
            Either "line", in which case a line will be plotted, or "scatter",
            in which case a scatter plot will be made. line is default.
        label : string, optional
            The label for the function. Default is ""
        color : string, optional
            Which color the plot should have. If None, a random colour is
            chosen.
        minimizing : bool, optional
            Whether the plot's goal is to minimize or maximize. Default is
            minimize.
    ax : Matplotlib.Axes, optional
        Axes to continue. If None, will return a new Ax and figure.
    fig_options : dict, optional
        Options used when creating a new plot.
        "legend_loc" : string, optional
            Location for the legend. Default is "upper right"
        "x_label" : string, optional
            x label for the figure
        "y_label" : string, optional
            y label for the figure
    plot_at_least : 2-float tuple, optional
        How many percent of the values should be displayed, from above and from
        below.
    plot_min : float, optional
        Plot from this value.
    plot_max : float, optional
        Plot up to this value.

    Returns
    -------
    fig : plt.figure
        Only when no ax had been specified.
    Ax : plt.Axes
        The plot containing the plotted lists.
    """
    fig = None
    if ax is None:
        fig, ax = create_figure(fig_options)

    for p in to_plot_list:
        if not "cutoff_percent" in p:
            continue
        if fig_options.get("minimizing", True):
            plot_min_this, plot_max_this = _get_y_min_max(p["y"], (
                1, p.get("cutoff_percent", 1)))
        else:
            plot_min_this, plot_max_this = _get_y_min_max(p["y"], (
                p.get("cutoff_percent", 1), 1))
        if plot_min is None or plot_min > plot_min_this:
            plot_min = plot_min_this
        if plot_max is None or plot_max > plot_max_this:
            plot_max = plot_max_this

    _plot_lists_ax(to_plot_list, ax, plot_min=plot_min, plot_max=plot_max)
    _polish_figure(ax, fig_options)

    if fig is None:
        return ax
    else:
        return fig, ax


def _plot_lists_ax(to_plot_list, ax, plot_min=None, plot_max=None):
    """
    Plots several functions.

    Each entry of to_plot_list defines x, y, and format options.

    Parameters
    ----------
    to_plot_list: list of dicts
        Defines the functions to plot.
        Each entry must contain at least values for "x" and
        "y", and can contain values for "type", "label" and "color".
        x: list
            A list of x values
        y: list
            A list of y values
        type="line": string
            Either "line", in which case a line will be plotted, or "scatter",
            in which case a scatter plot will be made.
        label="": string
            The label for the function.
        color=random colour: string
            Which color the plot should have.
    ax : Matplotlib.Axes
        Axes to continue.
    plot_at_least : 2-float tuple, optional
        How many percent of the values should be displayed, from above and from
        below.
    plot_min : float, optional
        Plot from this value.
    plot_max : float, optional
        Plot up to this value.

    Returns
    -------
    Ax : plt.Axes
        The plot containing the plotted lists.
    """
    for p in to_plot_list:
        ax = plot_single(p, ax, plot_min=plot_min, plot_max=plot_max)
    if plot_min is not None:
        ax.set_ylim(ymin=plot_min)
    if plot_max is not None:
        ax.set_ylim(ymax=plot_max)
    return ax


def _get_y_min_max(y, plot_at_least):
    """
    Returns the maximum / minimum values which should be plotted according to
    plot_at_least.

    Parameters
    ----------
    y : list of floats
        The y values in question.

    plot_at_least : 2-tuple of floats
        The (from_below, from_above) percentage of points to show.

    Returns
    -------
    min_y_new : float
        The new minimum y value.
    max_y_new : float
        The new maximum y value.
    """
    if len(y) == 0:
        return None, None
    sorted_y = sorted([x for x in y if str(x) != 'nan'])
    if plot_at_least[0] == 1:
        min_y_new = min(y)
    else:
        min_y_new = sorted_y[int(plot_at_least[0] * (-len(sorted_y)))]
    if plot_at_least[1] == 1:
        max_y_new = max(y)
    else:
        max_y_new = sorted_y[min(len(sorted_y)-1, int(plot_at_least[1] *
                                                      len(sorted_y)))]

    return min_y_new, max_y_new

COLORS = ["g", "r", "c", "b", "m", "y"]


def plot_single(to_plot, ax=None, fig_options=None, plot_min=None,
                plot_max=None):
    """
    Plots a single function.

    to_plot defines x, y, and format options.

    Parameters
    ----------
    to_plot : dict
        Defines the function to plot. Must contain at least values for "x" and
        "y", and can contain values for "type", "label" and "color".
        x : list
            A list of x values
        y : list
            A list of y values
        var : list
            A list of the variances for each y value. If exists, the resulting
            plot will have error bars.
        type : string, optional
            Either "line", in which case a line will be plotted, or "scatter",
            in which case a scatter plot will be made. Default is "line".
        label : string, optional
            The label for the function.
        color : string, optional
            Which color the plot should have.

    ax : pyplot.Axes, optional
        Axes to continue.
    fig_options : dict, optional
        Options used when creating a new plot.
        "legend_loc" : string, optional
            Location for the legend.
            Default is "upper right"
        "x_label" : string, optional
            x label for the figure
        "y_label" : string, optional
            y label for the figure

    Returns
    -------
    fig : plt.figure
        Either a new figure or fig, now containing the plots as specified.
    """
    if plot_max is not None and plot_max == plot_min:
        plot_min = plot_max - 0.1
    type = to_plot.get("type", "line")
    label = to_plot.get("label", None)
    color = to_plot.get("color", random.choice(COLORS))
    color = colorConverter.to_rgba(color)
    x = to_plot.get("x", [])
    y = to_plot.get("y", [])
    var = to_plot.get("var", [])
    #TODO Despite converting hte color to rgba, matplotlib does not want to change
    #the arrow colors. Cause is unclear.
    if type == "line":
        if "var" in to_plot:
            ax.errorbar(x, y, label=label, yerr=var, color=color,
                        linewidth=2.0, capthick=4, capsize=8.0)
        else:
            ax.plot(x, y, label=label, color=color, linewidth=2.0)
    elif type=="scatter":
        ax.scatter(x, y, label=label, color=color)
        if len(x) > 1 and max(y) != min(y):
            for i in range(len(x)):
                if y[i] is None:
                    continue

                if plot_min is None or plot_min is None:
                    arrow_len = 0.05
                else:
                    arrow_len = (plot_max - plot_min) * 0.05
                arrow_factor_min = abs(y[i] - min(y)) / (max(y) - min(y)) + 0.1
                arrow_factor_max = abs(max(y) - y[i]) / (max(y) - min(y)) + 0.1
                arrow_len_min = arrow_len * arrow_factor_min
                arrow_len_max = arrow_len * arrow_factor_max
                head_width = 0.2
                if plot_min is not None:
                    if plot_min > y[i]:
                        ax.arrow(x[i], plot_min+1.1*arrow_len_min, 0,
                                 -arrow_len_min/10, fc="k", ec="k",
                                 head_width=head_width,
                                 head_length=arrow_len_min,
                                 color=color)
                if plot_max is not None:
                    if plot_max < y[i]:
                        ax.arrow(x[i], plot_max-1.1*arrow_len_max, 0,
                                 arrow_len_max/10, fc="k", ec="k",
                                 head_width=head_width,
                                 head_length=arrow_len_max, color=color)
    return ax


def write_plot_to_file(fig, filename, store_path,  file_format="png",
                       transparent=False):
    """
    Write out plot to the file given in filename. Assumes that all
    directories already exist.

    Parameters
    ----------
    fig : matplotlib.figure
        The figure object to store.
    filename : string or os.path
        A string or path can be given here to specify where
        the plot is written to. All parent directories have to exist!
    file_format : string, optional
        Specifies file format of plot - all supported file formats
        by matplotlib can be given here. Default is "png"
    transparent : boolean, optional
        Specifies if a transparent figure is written. Default is False.
    """
    filename_w_extension = os.path.join(store_path, filename + "." +
                                        file_format)
    fig.savefig(filename_w_extension, format=file_format,
                transparent=transparent)


def create_figure(fig_options=None):
    """
    Creates a new figure with fig_options.

    Parameters
    ----------
    fig_options : dict, optional
        Options used when creating a new plot.
        "x_label" : string, optional
            x label for the figure
        "y_label" : string, optional
            y label for the figure
        "title" : string, optional
            The title for the figure.

    Returns
    -------
    fig : plt.figure
        A new figure with the options as specified in fig_options.
    """
    if fig_options is None:
        fig_options = {}
    fig, ax = plt.subplots()
    ax.set_xlabel(fig_options.get("x_label", ""))
    ax.set_ylabel(fig_options.get("y_label", ""))
    ax.set_title(fig_options.get("title", ""))
    return fig, ax


def _polish_figure(ax, fig_options=None):
    """
    Polishes a finished figure.

    Parameters
    ----------
    ax : matplotlib.Axes
        The ax to polish
    fig_options : dict, optional
        Options to be applied after the figure has been created. Supported:
        "legend_loc" : string
            Location for the legend. Default is "upper right"
    """
    if fig_options is None:
        fig_options = {}

    legend_loc = fig_options.get("legend_loc", "upper right")

    ax.set_xlim(left=0)
    if legend_loc == "no":
        # do nothing right now, since no legend
        pass
    else:
        ax.legend(loc=legend_loc)


