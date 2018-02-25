"""
    Numeric analysis methods
    Laboratory work number 2
    Variant 8, Zaharov Igor
"""

import math
import random
import scipy.optimize
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process


def method_of_chords(func, start: float, end: float, fault=0.00001):
    """
        Method of Chords
    :param func: function object, representing given function
    :param start: start border of segment
    :param end: end border of segment
    :param fault: погрешность
    :return: tuple - float root, number of iterations
    """
    # check incoming segments border is valid
    if end <= start or func(start) * func(end) > 0:
        raise ValueError('Borders is not correct')

    # select random point from segment
    x_rand = float(random.uniform(start, end))
    # check second derivative sign of func in randomized point
    second_derivative_is_negative = scipy.misc.derivative(func, x_rand, 2) < 0
    if second_derivative_is_negative:
        # select iterative function and x start value for convex up segment
        iterative_function = lambda x: start - func(start) / (func(x) - func(start)) * (x - start)
        x_n = end
    else:
        # select iterative function and x start value for convex down segment
        iterative_function = lambda x: x - func(x) / (func(end) - func(x)) * (end - x)
        x_n = start

    iteration_counter = 0
    while math.fabs(func(x_n)) > fault:
        # chords method step
        x_n = iterative_function(x_n)
        iteration_counter += 1

    return x_n, iteration_counter


def aitken_method(func, start: float, end: float, fault=0.00001):
    """
        Aitken method
        Note: Names for the variables are the same as a names in algorithm, demonstrated on the lecture
    :param func: function object, representing given function
    :param start: start border of segment
    :param end: end border of segment
    :param fault: погрешность
    :return: tuple - float root, number of iterations
    """
    # check incoming segments border is valid
    if end <= start or func(start) * func(end) > 0:
        raise ValueError('Borders is not correct')

    # find derivative local maximum
    func_derivative = lambda x: scipy.misc.derivative(func, x, 1)
    derivative_max = scipy.optimize.minimize_scalar(func_derivative,
                                                    bounds=(start, end), method='bounded')
    # construct iterative function fi(x) = x - 1/M * func(x)
    fi = lambda x: x - (1 / derivative_max.x) * func(x)
    # initial step for aitken method
    x_0 = float(random.uniform(start, end))
    x_1 = fi(x_0)
    x_2 = fi(x_1)
    x_temp = (x_0 * x_2 - x_1 ** 2) / (x_2 - 2 * x_1 + x_0)
    x_3 = fi(x_temp)

    # aitken method iterations
    iteration_counter = 0
    while math.fabs(func(x_3)) > fault:
        x_0 = x_temp
        x_2 = fi(x_1)
        x_1 = x_3

        x_temp = (x_0 * x_2 - x_1 ** 2) / (x_2 - 2 * x_1 + x_0)
        x_3 = fi(x_temp)
        iteration_counter += 1

    return x_3, iteration_counter


def stephenson_method(func, start: float, end: float, fault=0.00001):
    """
        Stephenson method
    :param func: function object, representing given function
    :param start: start border of segment
    :param end: end border of segment
    :param fault: погрешность
    :return: tuple - float root, number of iterations
    """
    # check incoming segments border is valid
    if end <= start or func(start) * func(end) > 0:
        raise ValueError('Borders is not correct')

    # select random value as initial
    x_n = float(random.uniform(start, end))
    iterative_function = lambda x: x - (func(x) ** 2) / (func(x) - func(x - func(x)))

    # stephenson method iterations
    iteration_counter = 0
    while math.fabs(func(x_n)) > fault:
        x_n = iterative_function(x_n)
        iteration_counter += 1

    return x_n, iteration_counter


def _function(x: float):
    """
        Function, given from the task
    :param x: argument
    :return: given function value
    """
    return x - 1 / math.atan(x)


def _display_plot(start, end, step):
    """ Display plot of _function(x) using pyplot """
    # make negative points range
    x_negative = np.arange(start, -0.1, step)
    y_negative = np.array([_function(item) for item in x_negative])
    # make positive points range
    x_positive = np.arange(0.1, end, step)
    y_positive = np.array([_function(x) for x in x_positive])

    # configure negative part plot and Y=0 line
    plt.plot(x_negative, y_negative, 'g')
    # configure positive part plot and Y=0 line
    plt.plot(x_positive, y_positive, 'g')
    # configure lines
    lines_step = 0.5
    # line X=0 configure
    x_range = np.arange(start, end, step)
    plt.plot(x_range, np.zeros(len(x_range)), 'b')
    # line Y=0 configure
    y_range = np.arange(-10, 10, lines_step)
    plt.plot(np.zeros(len(y_range)), y_range, 'b')

    # disable interactive mod and show plot
    plt.ioff()
    plt.show()


def main():
    """ User interaction logic """
    # plot compact borders
    l, r, step = -2, 2, 0.001
    plt_window = Process(target=_display_plot, args=(l, r, step))
    plt_window.start()

    # event loop, can be interrupt by KeyboardInterrupt Ctrl+C
    while True:
        # user input
        print('\nInput [start, end] of segment, where one root exist\n')
        start, end = float(input('start: ')), float(input('end: '))

        method_of_chords_res = method_of_chords(_function, start, end)
        aitken_method_res = aitken_method(_function, start, end)
        stephenson_method_res = stephenson_method(_function, start, end)
        print('method of chords: root {}, iter {}'.format(*method_of_chords_res))
        print('aitken method: root {}, iter {}'.format(*aitken_method_res))
        print('stephenson method: root {}, iter {}'.format(*stephenson_method_res))


if __name__ == '__main__':
    main()
