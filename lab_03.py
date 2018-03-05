"""
    Numeric analysis methods
    Laboratory work number 3
    Variant 8, Zaharov Igor
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process


def _bernoulli_method_coefficients(polynom):
    """
        Bernoulli method
    Main proposal of the method is to isolate segments, where root exists
    Variables names are the same, as in the algorithm from the lecture
    :param polynom: list of polynomial coefficients, 0-s index is for the biggest power and so
    :return: tuple of four special values for isolating roots
    """
    # biggest absolute value of initial polynomial negative coefficients
    b0 = math.fabs(min(filter(lambda a: a < 0, polynom)))
    # special value for the bernoulli method
    n0 = 1 + (b0 / math.fabs(polynom[0]))

    # P1(x) polynomial in algorithm: reversed order
    p1 = list(reversed(polynom))
    # biggest absolute value of p1 negative coefficients
    b1 = math.fabs(min(filter(lambda a: a < 0, p1)))
    n1 = 1 + math.sqrt(b1 / math.fabs(p1[0]))

    # P2(x) polynomial in algorithm: every item, except a_n, multiplied by -1^n (n - power of X)
    p2 = [(-1)**(n - 1) * polynom[-n] for n in range(len(polynom), 0, -1)]
    # similarly as with p1
    b2 = math.fabs(min(filter(lambda a: a < 0, p2)))
    n2 = 1 + (b2 / math.fabs(p2[0])) ** (1. / 3)

    # P3(x) polynomial in algorithm: reversed p2 with opposed sign's
    p3 = list(reversed(list(map(lambda a: -1 * a, p2))))
    b3 = math.fabs(min(filter(lambda a: a < 0, p3)))
    n3 = 1 + (b3 / math.fabs(p3[0])) ** (1. / 4)

    return n0, n1, n2, n3


def bernoulli_method(polynom):
    """
        Bernoulli method
    :param polynom: ist of polynomial coefficients, 0-s index is for the biggest power and so on
    :return: dict with keys 'positive' 'negative' and values - tuples with segment borders
    """
    n = list(_bernoulli_method_coefficients(polynom))

    segments = dict()
    segments['positive'] = 1 / n[1], n[0]
    segments['negative'] = - n[2], -1. / n[3]

    return segments


def _function(x):
    """
        Function, given from the task
    :param x: function argument
    :return: function value
    """
    return x**3 - 3 * x**2 + 2.5


def _display_plot(start, end, step):
    """ Display plot of _function(x) using pyplot """
    # make points range
    x = np.arange(start, end, step)
    y = np.array([_function(item) for item in x])

    # configure _function plot and Y = 0
    plt.plot(x, y)
    plt.plot(x, np.zeros(len(x)), color='red', linewidth=0.4, linestyle='--')

    # disable interactive mode and show plot
    plt.ioff()
    plt.show()


def main():
    """ Main logic """
    # plot compact borders
    l, r, step = -3, 3, 0.0001
    plt_window = Process(target=_display_plot, args=(l, r, step), )
    plt_window.start()

    polynom = [1, -3, 0, 2.5, ]
    root_segments = bernoulli_method(polynom)

    print('positive: {}  to  {}'.format(*root_segments['positive']))
    print('negative: {} to {}'.format(*root_segments['negative']))

    plt_window.join()


if __name__ == '__main__':
    main()
