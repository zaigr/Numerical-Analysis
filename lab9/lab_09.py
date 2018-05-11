"""
    Numeric analysis methods
    Laboratory work number 9
    Variant 8, Zaharov Igor

    Least-squares function approximation
"""

import json
import math
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process


def least_square_exponential(x_n, y_n):
    """
        Least-squares function approximation
        by exponential function a * exp(b*x)
    :param x_n: interpolation nodes, args
    :param y_n: interpolation nodes, values
    :return: function representing interpolated polynomial
    """
    sys_matrix = np.array([[sum(x ** 2 for x in x_n), sum(x_n)],
                           [sum(x_n),                 len(x_n)]])
    solution_row = np.array([sum(x_n[i] * math.log(y_n[i]) for i in range(len(x_n))),
                             sum(math.log(y) for y in y_n)])
    a, B = np.linalg.solve(sys_matrix, solution_row)
    b = math.exp(B)

    def interpolation(x):
        """ Returns interpolated function value """
        return a * math.exp(b * x)

    return interpolation


def least_square_log(x_n, y_n):
    """
        Least-squares function approximation
        by logarithmic function a * ln(x) + b
    :param x_n: interpolation nodes, args
    :param y_n: interpolation nodes, values
    :return: function representing interpolated polynomial
    """
    sys_matrix = np.array([[sum(math.log(x) ** 2 for x in x_n), sum(math.log(x) for x in x_n)],
                           [sum(math.log(x) for x in x_n),      len(x_n)]])
    solution_row = np.array([sum(y_n[i] * math.log(x_n[i]) for i in range(len(x_n))),
                             sum(y_n)])
    a, b = np.linalg.solve(sys_matrix, solution_row)

    def interpolation(x):
        """ Returns interpolated function value """
        return a * math.log(x) + b

    return interpolation


def least_square_hyperbolic(x_n, y_n):
    """
        Least-squares function approximation
        by hyperbolic function a / x + b
    :param x_n: interpolation nodes, args
    :param y_n: interpolation nodes, values
    :return: function representing interpolated polynomial
    """
    sys_matrix = np.array([[sum(1.0 / (x ** 2) for x in x_n), sum(1.0 / x for x in x_n)],
                           [sum(1.0 / x for x in x_n),                 len(x_n)]])
    solution_row = np.array([sum(y_n[i] / x_n[i] for i in range(len(x_n))),
                             sum(y_n)])
    a, b = np.linalg.solve(sys_matrix, solution_row)

    def interpolation(x):
        """ Returns interpolated function value """
        return a / x + b

    return interpolation


def least_square_linear(x_n, y_n):
    """
        Least-squares function approximation
        by linear function a*x + b = 0
    :param x_n: interpolation nodes, args
    :param y_n: interpolation nodes, values
    :return: function representing interpolated polynomial
    """
    sys_matrix = np.array([[sum(x ** 2 for x in x_n), sum(x_n)],
                           [sum(x_n),                 len(x_n)]])
    solution_row = np.array([sum(x_n[i] * y_n[i] for i in range(len(x_n))),
                             sum(y_n)])
    a, b = np.linalg.solve(sys_matrix, solution_row)

    def interpolation(x):
        """ Returns interpolated function value """
        return a * x + b

    return interpolation


def least_square_quadratic(x_n, y_n):
    """
        Least-squares function approximation
        by quadratic polynomial a*x^2 + b*x + c = 0
    :param x_n: interpolation nodes, args
    :param y_n: interpolation nodes, values
    :return: function representing interpolated polynomial
    """
    # the method reduces to solving system
    # which was a result of finding the minimum
    # of the function a*x^2 + b*x + c = 0 by variables
    # 'a', 'b' and 'c'
    sys_matrix = np.array([[sum(x ** 4 for x in x_n), sum(x ** 3 for x in x_n), sum(x ** 2 for x in x_n)],  # df/da
                           [sum(x ** 3 for x in x_n), sum(x ** 2 for x in x_n), sum(x_n)],                  # df/db
                           [sum(x ** 2 for x in x_n), sum(x_n), len(x_n)]])                                 # df/dc
    solution_row = np.array([sum((x_n[i] ** 2) * y_n[i] for i in range(0, len(x_n))),
                             sum(x_n[i] * y_n[i] for i in range(0, len(x_n))),
                             sum(y_n)])
    a, b, c = np.linalg.solve(sys_matrix, solution_row)

    def interpolation(x):
        """ Returns interpolated function value """
        return a * (x ** 2) + b * x + c

    return interpolation


def _display_plot(x_range, y_range, x_nodes, y_nodes):
    """
        Display plot of interpolated values using pyplot
    :param x_range: range of arguments, had interpolated
    :param y_range: range of interpolated values
    :param x_nodes: interpolation nodes args
    :param y_nodes: interpolation nodes values
    """
    plt.plot(x_range, y_range, label='interpolation')  # interpolated function
    plt.plot(x_nodes, y_nodes,                                  # interpolation nodes
             'o', linewidth=0.01, label='interpolation nodes')

    plt.ioff()              # disable window interactive mode
    plt.legend(loc='best')  # enable labels in plot
    plt.show()


def _display_plot_async(x_range, y_range, x_nodes, y_nodes):
    """
        Run function _display_plot in another process
    :param x_range:
    :param y_range:
    :param x_nodes:
    :param y_nodes:
    :return:
    """
    process = Process(target=_display_plot, args=(x_range, y_range, x_nodes, y_nodes))
    process.start()


def main():
    """ user interaction logic """
    with open('points.json', 'r') as file:
        data = json.load(file)
        x_nodes = data['x']
        y_nodes = data['y']

    # in this case, it's more effective to use quadratic approximation
    interpolation = least_square_quadratic(x_nodes, y_nodes)

    x_range = np.arange(x_nodes[0], x_nodes[-1], 0.01)
    y_range = np.array([interpolation(x) for x in x_range])

    _display_plot_async(x_range, y_range, x_nodes, y_nodes)

    # another cases of interpolation
    # 1) linear function
    linear = least_square_linear(x_nodes, y_nodes)
    y_range = np.array([linear(x) for x in x_range])
    _display_plot_async(x_range, y_range, x_nodes, y_nodes)

    # 2) hyperbolic function
    hyperbolic = least_square_hyperbolic(x_nodes, y_nodes)
    y_range = np.array([hyperbolic(x) for x in x_range])
    _display_plot_async(x_range, y_range, x_nodes, y_nodes)

    # 3) logarithmic function
    logarithmic = least_square_log(x_nodes, y_nodes)
    y_range = np.array([logarithmic(x) for x in x_range])
    _display_plot_async(x_range, y_range, x_nodes, y_nodes)

    # 4) exponential function
    exponential = least_square_exponential(x_nodes, y_nodes)
    y_range = np.array([exponential(x) for x in x_range])
    _display_plot_async(x_range, y_range, x_nodes, y_nodes)


if __name__ == '__main__':
    main()
