"""
    Numeric analysis methods
    Laboratory work number 9
    Variant 8, Zaharov Igor

    Cubic spline interpolation
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


def _display_plot(x_nodes, y_nodes, x_range, y_range):
    """
        Display plot of interpolated values using pyplot
    :param x_nodes: interpolation nodes args
    :param y_nodes: interpolation nodes values
    :param x_range: range of arguments, had interpolated
    :param y_range: range of interpolated values
    :return:
    """
    plt.plot(x_nodes, y_nodes, 'o', label='interpolation nodes')
    plt.plot(x_range, y_range, label='interpolation')

    plt.ioff()              # disable window interactive mode
    plt.legend(loc='best')  # enable labels in plot
    plt.show()


def main():
    """ user interaction logic """
    with open('points.json', 'r') as file:
        data = json.loads(file.read())
        x_n = data['x']
        y_n = data['y']

    # interpolation = spline_interpolation(x_n, y_n)
    # x_range = np.arange(x_n[0], x_n[-1], 0.01)
    # y_range = np.array([interpolation(x) for x in x_range])

    cs = CubicSpline(x_n, y_n)

    x_range = np.arange(x_n[0], x_n[-1], 0.1)
    y_range = cs(x_range)

    plt.plot(x_range, cs(x_range, 1), label="S'", linestyle='--')
    plt.plot(x_range, cs(x_range, 2), label="S''", linestyle='--')
    plt.plot(x_range, cs(x_range, 3), label="S'''", linestyle='--')

    _display_plot(x_n, y_n, x_range, y_range)


if __name__ == '__main__':
    main()
