"""
    Numeric analysis methods
    Laboratory work number 6
    Variant 8, Zaharov Igor

    Newton interpolation polynomial
"""

import json
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple


Compact = namedtuple('Compact', ['start', 'end', 'step'])


def newton_interpolation(x, y):
    """
        Newton interpolation polynomial building
    :param x: range of interpolation arguments
    :param y: range of interpolation values
    :return: function representing interpolated polynomial
    """
    f = [y, ]  # divided difference table

    # divided difference table building
    for step in range(1, len(x)):
        differences = list()
        for i in range(1, len(f[step - 1])):
            differ = (f[step - 1][i] - f[step - 1][i - 1]) / (x[i + step - 1] - x[i - 1])
            differences.append(differ)
        f.append(differences)

    def polynomial(arg):
        """ Returns interpolated polynomial value """
        polynom_val = y[0]
        x_product = 1
        for i in range(1, len(f)):
            x_product *= arg - x[i - 1]
            polynom_val += f[i][0] * x_product
        return polynom_val

    return polynomial


def display_plot(polynom, compact, x_interp, y_interp, args, values):
    """ Display plot of polynom(x) using pyplot """
    # make points range
    x = np.arange(*compact)
    y = np.array([polynom(item) for item in x])

    plt.plot(x, y, label='interpolation')               # polynomial plot
    plt.plot(x, np.zeros(len(x)), linewidth=0.3,        # line Y=0 plot
             linestyle='--', color='red', label='Y=0')
    plt.plot(args, values, 'bo', label='values')        # counted values plot

    # display interpolation nodes
    plt.plot(x_interp, y_interp, 'o', linewidth=0.01,
             label='interpolation nodes')

    # disable interactive mode and show plot
    plt.ioff()
    plt.legend(loc='best')
    plt.show()


def main():
    """ Main logic """
    # read points range from json file
    with open('points.json', 'r') as file:
        point_ranges = json.loads(file.read())
        x = point_ranges['x']
        y = point_ranges['y']
        args = point_ranges['args']

    interpolation = newton_interpolation(x, y)

    values = []
    for arg in args:
        value = interpolation(arg)
        values.append(value)

    compact = Compact(start=0, end=0.9, step=0.001)
    display_plot(interpolation, compact, x, y, args, values)


if __name__ == '__main__':
    main()
