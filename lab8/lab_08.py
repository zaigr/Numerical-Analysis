"""
    Numeric analysis methods
    Laboratory work number 9
    Variant 8, Zaharov Igor

    Cubic spline interpolation
"""

import json
import bisect
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


def parabolic_spline(x_nodes, y_nodes):
    """
        Parabolic spline interpolation
    :param x_nodes: interpolation nodes arguments
    :param y_nodes: interpolation nodes values
    :return: function, representing interpolated polynomial
    """
    a = y_nodes

    z = [2 * (y_nodes[i + 1] - y_nodes[i]) / (x_nodes[i + 1] - x_nodes[i])
         for i in range(len(y_nodes) - 1)]

    b = [0, ] * (len(y_nodes))  # init empty list b
    b[0] = a[0]
    for i in range(len(y_nodes)):
        b[i] = z[i - 1] - b[i - 1]

    c = [0, ] * (len(y_nodes) - 1)
    for i in range(len(y_nodes) - 1):
        c[i] = (b[i + 1] - b[i]) / (2 * (x_nodes[i + 1] - x_nodes[i]))

    def interpolation(x):
        """
            Interpolated function
        :param x: interpolated function argument
        :return: interpolated function value
        """
        insert_idx = bisect.bisect_left(x_nodes, x)
        if x < x_nodes[insert_idx]:
            return a[insert_idx - 1] + (x - x_nodes[insert_idx - 1]) * b[insert_idx - 1] + \
                    + c[insert_idx - 1] * (x - x_nodes[insert_idx - 1]) ** 2
        else:
            return a[insert_idx]

    return interpolation


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

    ps = parabolic_spline(x_n, y_n)

    x_range = np.arange(x_n[0], x_n[-1], 0.001)
    y_range = [ps(x) for x in x_range]

    _display_plot(x_n, y_n, x_range, y_range)


if __name__ == '__main__':
    main()
