"""
    Multiprocess way to display range of points
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process


def display_plot_async(x_range, y_range):
    """
        Run function _display_plot in another process
    :param x_range:
    :param y_range:
    :param x_nodes:
    :param y_nodes:
    :return:
    """
    process = Process(target=_display_plot, args=(x_range, y_range))
    process.start()


def _display_plot(x_range, y_range):
    """
        Display plot of interpolated values using pyplot
    :param x_range: range of arguments, had interpolated
    :param y_range: range of interpolated values
    :param x_nodes: interpolation nodes args
    :param y_nodes: interpolation nodes values
    """
    plt.plot(x_range, y_range, label='interpolation')  # interpolated function

    plt.ioff()              # disable window interactive mode
    plt.legend(loc='best')  # enable labels in plot
    plt.show()

