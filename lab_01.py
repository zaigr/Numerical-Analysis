"""
    Numeric analysis methods
    Laboratory work number 1
    Variant 8, Zaharov Igor
"""

import math
import random
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
    # check incoming segments border is valid
    if end <= start or func(start) * func(end) > 0:
        raise ValueError('Borders is not correct')

    # find derivative local maximum
    def _func_derivative(x):
        return (func(x + fault) - func(x)) / ((x + fault) - x)
    local_func_max = optimize.minimize_scalar(_func_derivative,
                                              bounds=(start, end), method='bounded')
    # represent func as g(x) = x - (1 / M) * func(x)
    if local_func_max.fun > 0:
        iterative_func = lambda c: c - (1 / local_func_max.x) * func(c)
    else:
        # case when derivative is negative
        iterative_func = lambda c: c + (1 / local_func_max.x) * func(c)

    iteration_counter = 0
    # select random x0 from segment
    x_0 = float(random.uniform(start, end))
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
    # check incoming segments border is valid
    if end <= start or func(start) * func(end) > 0:
        raise ValueError('Borders is not correct')

    iteration_counter = 0
    # split initial segment into two parts
    x = (start + end) / 2
    while math.fabs(func(x)) > fault:
        # case: root is on the left hand side or in right
        if func(start) * func(x) < 0:
            end = x
        else:
            start = x
        # again split new segment into two parts
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
    # check incoming segments border is valid
    if not start < start_x < end or func(start) * func(end) > 0:
        raise ValueError('Borders is not correct')

    # using approximated derivative by the algorithm
    def approx_func_derivative(c):
        return (func(start_x) - func(c)) / (start_x - c)

    iteration_counter = 0
    # select random x from given segment
    x = float(random.uniform(start, end))
    while math.fabs(func(x)) > fault:
        # Newtons simple method step
        x = x - func(x) / approx_func_derivative(x)
        iteration_counter += 1

    return x, iteration_counter


def _display_plot(start, end, step):
    """ Display plot of _function(x) using pyplot """
    # make points range
    x = np.arange(start, end, step)                # make range from [0] to [1] with step [2]
    y = np.array([_function(item) for item in x])  # convert list to nparray object

    # configure _function plot and Y=0 line
    plt.plot(x, y)
    plt.plot(x, np.zeros(len(x)))

    # disable interactive mod and show plot
    plt.ioff()
    plt.show()


def _function(x: float):
    """ Function, given from the task """
    return 2 * math.log10(x) - x / 2 + 1


def main():
    """ User interaction logic """
    # plot compact borders
    l, r, step = 0.01, 20, 0.001
    # start pyplot window process
    plt_window = Process(target=_display_plot, args=(l, r, step), )
    plt_window.start()

    # event loop, can be interrupt by KeyboardInterrupt Ctrl+C
    while True:
        try:
            # users input
            print('\nInput [start, end] of segment, where one root exist\n')
            start, end = float(input('start: ')), float(input('end: '))
            start_x = float(input('input x, near the root: '))

            # methods executing
            bisection_res = bisection_method(_function, start, end)
            simple_newton_res = newton_simple_method(_function, start, end, start_x)
            iteration_res = simple_iteration_method(_function, start, end)
            print('bisection method: root {}, iter {}'.format(*bisection_res))
            print('newton simple method: root {}, iter {}'.format(*simple_newton_res))
            print('simple iteration method root {}, iter {}'.format(*iteration_res))
        except Exception as e:
            print('Exception occurred! ' + str(e))

if __name__ == '__main__':
    main()
