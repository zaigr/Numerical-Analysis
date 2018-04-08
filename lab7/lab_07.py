"""
    Numeric analysis methods
    Laboratory work number 7
    Variant 8, Zaharov Igor

    Lagrange interpolation polynomial and
    Aitken's scheme
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

# tuple representing point with coordinate X and Y
Point = namedtuple('Point', ['x', 'y'])


def aitken_interpolation(arg, x_nodes, y_nodes):
    """
        Aitken interpolation method
    :param arg: argument of interpolated polynomial
    :param x_nodes: range of interpolation arguments
    :param y_nodes: range of interpolation values
    :return: value of interpolation polynomial
    """
    table = _divided_difference(arg, x_nodes, y_nodes)  # divided difference table

    # return last divided difference of the table
    return table[-1][0]


def _divided_difference(arg, x, y):
    """
        Returns divided difference table for
        Aitken interpolation method
    :param arg: argument of interpolated polynomial
    :param x: range of interpolation arguments
    :param y: range of interpolation values
    :return: interpolated value of polynomial
    """
    f = [y, ]  # divided difference table
    for step in range(1, len(y)):
        differences = list()
        for i in range(1, len(f[step - 1])):
            matrix = np.array([[f[step - 1][i - 1], x[i - 1] - arg],
                               [f[step - 1][i], x[i + step - 1] - arg]])
            polynom = 1 / (x[i + step - 1] - x[i - 1]) * np.linalg.det(matrix)
            differences.append(polynom)
        f.append(differences)

    return f


def lagrange_interpolation(x, y):
    """
        Lagrange interpolation polynomial
    :param x: range of interpolation arguments
    :param y: range of interpolation values
    :return: function representing interpolated polynomial
    """
    def polynomial(arg):
        """ Returns interpolated polynomial value """
        value = 0
        for i in range(0, len(y)):
            value += y[i] * _l(i, arg, x)
        return value

    return polynomial


def _l(index, x, x_nodes):
    """
        Basic polynomial counting
    :param index: the order of basic polynomial
    :param x: polynomial argument
    :param x_nodes: range of interpolation arguments
    :return: value of basic polynomial
    """
    val = 1
    for i in range(0, len(x_nodes)):
        if i != index:
            val *= (x - x_nodes[i]) / (x_nodes[index] - x_nodes[i])
    return val


def _display_plot(x, y, x_nodes, y_nodes, point):
    """
        Display plot of polynom(x) using pyplot
    :param x, y: points of polynomial plot
    :param x_nodes, y_nodes: interpolation nodes
    :param point: tuple of type Point
    """
    plt.plot(x, y, label='interpolation')               # polynomial plot
    plt.plot(x_nodes, y_nodes, 'o', linewidth=0.01,     # interpolation points
             label='interpolation nodes')
    plt.plot(x, np.zeros(len(x)), linewidth=0.3,        # line Y=0 plot
             linestyle='--', color='red', label='Y=0')
    plt.plot(point.x, point.y, 'bo', label='aitken')    # counted point plot

    plt.ioff()              # disable window interactive mode
    plt.legend(loc='best')  # enable labels in plot
    plt.show()


def main():
    """ Main logic """
    with open('points.json', 'r') as file:
        points = json.loads(file.read())
        x_nodes = points['x']
        y_nodes = points['y']
        arg = points['arg']

    interpolation = lagrange_interpolation(x_nodes, y_nodes)
    value = aitken_interpolation(arg, x_nodes, y_nodes)

    x_range = np.arange(0, 0.9, 0.001)
    y_range = np.array([interpolation(arg) for arg in x_range])
    point = Point(x=arg, y=value)

    _display_plot(x_range, y_range, x_nodes, y_nodes, point)


if __name__ == '__main__':
    main()
