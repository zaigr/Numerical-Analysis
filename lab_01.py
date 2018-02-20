"""
    Numeric analysis methods
    Laboratory work number 1
    Variant 8, Zaharov Igor
"""

import math
import scipy
import sys
import random
import sympy
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from multiprocessing import Process


def simple_iteration_method(func, start, end, fault=0.00001):
    """
        Simple iteration method
    :param func: function object, representing given function
    :param start: left border of segment
    :param end: right border of segment
    :param fault: погрешность
    :return: tuple - float root, number of iterations
    """
    # make func derivative
    # P.S. dont assign to lambda!!
    func_derivative = lambda x: (func(x + fault) - func(x)) / ((x + fault) - x)
    # find derivative local maximum
    local_func_max = optimize.minimize_scalar(func_derivative,
                                              bounds=(start, end), method='bounded')
    # represent func as g(x) = x - (1 / M) * func(x)
    if local_func_max.fun > 0:
        iterative_func = lambda c: c - (1 / local_func_max.x) * func(c)
    else:
        # case when derivative is negative
        iterative_func = lambda c: c + (1 / local_func_max.x) * func(c)

    iteration_counter = 0
    x_0 = random.uniform(start, end)
    while math.fabs(func(x_0)) > fault:
        x_0 = iterative_func(x_0)
        iteration_counter += 1

    return x_0, iteration_counter


def bisection_method(func, start, end, fault=0.00001):
    """
        Bisection method
    :param func: function object, representing given function
    :param start: start border of segment
    :param end: end border of segment
    :param fault: погрешность
    :return: tuple -  float root, number of iterations
    """
    if start < end and func(start) * func(end) > 0:
        raise ValueError('Borders is not correct')

    iteration_counter = 0
    x = (start + end) / 2
    while math.fabs(func(x)) > fault:
        if func(start) * func(x) < 0:
            end = x
        else:
            start = x
        x = (start + end) / 2
        iteration_counter += 1

    return x, iteration_counter


def newton_simple_method(func, start, end, start_x, fault=0.00001):
    """
        Newton simple method
    :param func: function object, representing given function
    :param start: start border of segment
    :param end: end border of segment
    :param start_x: value that is near the root
    :param fault: погрешность
    :return: tuple - float root, number of iterations
    """
    if start < end and func(start) * func(end) > 0:
        raise ValueError('Borders is not correct')

    iteration_counter = 0
    func_derivative = lambda c: (func(start_x) - func(c)) / (start_x - c)
    x = float(random.uniform(start, end))
    while math.fabs(func(x)) > fault:
        x = x - func(x) / func_derivative(x)
        iteration_counter += 1

    return x, iteration_counter


def main():
    """
        Main logic
        give a command line argument i to on interactive matplotlib mode
    """
    # given function from task
    given_function = lambda x: 2 * math.log10(x) - x / 2 + 1

    # plot making
    x_values = np.arange(0.01, 40, 0.01)                         # make range from [0] to [1] with step [2]
    y_values = np.array([given_function(x) for x in x_values])   # convert list to nparray object

    plt.plot(x_values, y_values, label='2 * lg(x) - x / 2 + 1')  # main plot setting, not necessary to use nparrays
    plt.plot(x_values, np.zeros(len(x_values)), label='Y')       # y=0 line setting

    # TODO: Make pyplot window in another process
    if 'i' in sys.argv:  # enable matplotlib interactive mode
        plt.ion()        # by default interactive mode is disabled
    plt.show()           # remember about GIL, use multiprocessing to make window clear interactive

    # event loop, can be interrupt by KeyboardInterrupt Ctrl+C
    while True:
        # users input
        print('\nInput [start, end] of segment, where one root exist\n')
        start, end = float(input('start: ')), float(input('end: '))
        start_x = float(input('input x, near the root: '))

        # methods executing
        bisection_res = bisection_method(given_function, start, end)
        simple_newton_res = newton_simple_method(given_function, start, end, start_x)
        iteration_res = simple_iteration_method(given_function, start, end)
        print('bisection method: root {}, iter {}'.format(*bisection_res))
        print('newton simple method: root {}, iter {}'.format(*simple_newton_res))
        print('simple iteration method root {}, iter {}'.format(*iteration_res))


if __name__ == '__main__':
    main()
