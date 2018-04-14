"""
    Numeric analysis methods
    Laboratory work number 6
    Variant 8, Zaharov Igor

    Newton interpolation polynomial
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process


def newton_interpolation_straight(x, y):
    """
        Newton interpolation polynomial building
        using straight interpolation formula
    :param x: range of interpolation arguments
    :param y: range of interpolation values
    :return: function representing interpolated polynomial
    """
    # divided difference table
    f = _divided_difference(x, y)

    def polynomial(arg):
        """ Returns interpolated polynomial value """
        polynom_val = y[0]
        x_product = 1
        for i in range(1, len(f)):
            x_product *= arg - x[i - 1]
            polynom_val += f[i][0] * x_product
        return polynom_val

    return polynomial


def newton_interpolation_inverse(x, y):
    """
        Newton interpolation polynomial building
        using inverse interpolation formula
    :param x: range of interpolation arguments
    :param y: range of interpolation values
    :return: function representing interpolated polynomial
    """
    # divided difference table
    f = _divided_difference(x, y)

    def polynomial(arg):
        """ Returns interpolated polynomial value """
        polynom_val = y[-1]
        x_product = 1
        for i in range(1, len(f)):
            x_product *= arg - x[-i]
            polynom_val += f[i][-1] * x_product
        return polynom_val

    return polynomial


def _divided_difference(x, y):
    """
        Returns a divided difference table
        for newton interpolation method
    :param x: range of interpolation arguments
    :param y: range of interpolation values
    :return: list of lists
    """
    f = [y, ]  # divided difference table

    # divided difference table building
    for step in range(1, len(x)):
        differences = list()
        for i in range(1, len(f[step - 1])):
            difference = (f[step - 1][i] - f[step - 1][i - 1]) / (x[i + step - 1] - x[i - 1])
            differences.append(difference)
        f.append(differences)

    return f


def display_plot(
        x, y,                # interpolated polynomial points
        x_interp, y_interp,  # interpolation nodes
        args, values         # values from the task
        ):
    """ Display plot of polynom(x) using pyplot """
    plt.plot(x, y, label='interpolation')               # polynomial plot
    plt.plot(x, np.zeros(len(x)), linewidth=0.3,        # line Y=0 plot
             linestyle='--', color='red', label='Y=0')
    plt.plot(args, values, 'bo', label='values')        # counted values plot

    # display interpolation nodes
    plt.plot(x_interp, y_interp, 'o', linewidth=0.01,
             label='interpolation nodes')

    # disable interactive mode
    plt.ioff()
    plt.legend(loc='best')
    plt.show()


def _process_interpolation(interpolation, x, y, args):
    """
        Logic of task executing and plot building
    :param x: range of interpolation arguments
    :param y: range of interpolation values
    :param args: arguments, given from the task
    :return: None
    """
    # line of polynomial plot setup
    start, end, step = 0, 0.9, 0.001
    x_range = np.arange(start, end, step)
    y_range = np.array([interpolation(arg) for arg in x_range])

    # points, should be counted by the task
    values = []
    for arg in args:
        value = interpolation(arg)
        values.append(value)

    # run pyplot window in another process
    plot_args = x_range, y_range, x, y, args, values
    window = Process(target=display_plot, args=plot_args)
    window.start()


def main():
    """ Main logic """
    # read points range from json file
    with open('points.json', 'r') as file:
        point_ranges = json.loads(file.read())
        x = point_ranges['x']
        y = point_ranges['y']
        args = point_ranges['args']

    str_interpolation = newton_interpolation_straight(x, y)
    _process_interpolation(str_interpolation, x, y, args)

    inv_interpolation = newton_interpolation_inverse(x, y)
    _process_interpolation(inv_interpolation, x, y, args)


if __name__ == '__main__':
    main()
