__author__ = 'Frederik Diehl'
import matplotlib.pyplot as plt
import random

def plot_lists(to_plot_list, fig=None, fig_options=None):
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
    fig=None: pyplot.figure
        A plot to continue, or None in which case a new plot is made using
        plot_options.
    fig_options=None: dict
        Options used when creating a new plot.
        "legend_loc"="upper right": string
            Location for the legend.
        "x_label"="": string
            x label for the figure
        "y_label"="": string
            y label for the figure

    Returns
    -------
    fig: plt.figure
        Either a new figure or fig, now containing the plots as specified.
    """
    newly_created = False
    if fig is None:
        fig = _create_figure(fig_options)
        newly_created = True
    for p in to_plot_list:
        fig = plot_single(p, fig)

    if newly_created:
        _polish_figure(fig_options)

    return fig

COLORS = ["g", "r", "c", "b", "m", "y"]


def plot_single(to_plot, fig=None, fig_options=None):
    """
    Plots a single function.

    to_plot defines x, y, and format options.

    Parameters
    ----------
    to_plot: dict
        Defines the function to plot. Must contain at least values for "x" and
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
    fig=None: pyplot.figure
        A plot to continue, or None in which case a new plot is made using
        fig_options.
    fig_options=None: dict
        Options used when creating a new plot.
        "legend_loc"="upper right": string
            Location for the legend.
        "x_label"="": string
            x label for the figure
        "y_label"="": string
            y label for the figure

    Returns
    -------
    fig: plt.figure
        Either a new figure or fig, now containing the plots as specified.
    """
    newly_created = False
    if fig is None:
        fig = _create_figure(fig_options)
        newly_created = True
    plt.figure(fig.number)
    type = to_plot.get("type", "line")
    label = to_plot.get("label", "")
    color = to_plot.get("color", random.choice(COLORS))
    x = to_plot.get("x", [])
    y = to_plot.get("y", [])
    print("label: %s" %label)

    if type == "line":
        plt.plot(x, y, label=label, color=color)
    elif type=="scatter":
        plt.scatter(x, y, label=label, color=color)

    if newly_created:
        _polish_figure(fig_options)
    return fig

def _create_figure(fig_options=None):
    """
    Creates a new figure with fig_options.

    Parameters
    ----------
    fig_options=None: dict
        Options used when creating a new plot.
        "x_label"="": string
            x label for the figure
        "y_label"="": string
            y label for the figure
        "title"="": string
            The title for the figure.

    Returns
    -------
    fig: plt.figure
        A new figure with the options as specified in fig_options.
    """
    if fig_options is None:
        fig_options = {}
    fig = plt.figure()
    plt.xlabel(fig_options.get("x_label", ""))
    plt.ylabel(fig_options.get("y_label", ""))
    plt.title(fig_options.get("title", ""))
    return fig

def _polish_figure(fig_options=None):
    """
    Polishes a finished figure.

    fig_options=None: dict
        Options used.
        "legend_loc"="upper right": string
            Location for the legend.
    """
    if fig_options is None:
        fig_options = {}
    plt.legend(loc=fig_options.get("legend_loc", "upper right"))