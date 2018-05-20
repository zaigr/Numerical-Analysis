"""
    Numeric analysis methods
    Laboratory work number 10
    Variant 8, Zaharov Igor

    Numerical integration, Task 2, execution logic
"""

from main_1 import trapeze_integration, simpson_integration
import math
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process


t_n = {
    3: [0.707107, 0, -0.707107],
    4: [0.794654, 0.187592, -0.187592, -0.794654],
}

c_n = {
    3: 2.0 / 3,
    4: 0.5
}


def chebyshev_integration(function, start, end, n):
    """

    :param func:
    :param start:
    :param end:
    :param n:
    :return:
    """
    n = 4
    coefs = t_n[n]
    x_n = []
    for t in coefs:
        x = (start + end) / 2 + ((end - start) / 2) * t
        x_n.append(x)

    sum = 0
    for x in x_n:
        sum += function(x)

    integral = ((end - start) / n) * sum
    return integral


def _display_plot_async(x_range, y_range):
    """
        Run function _display_plot in another process
    :param x_range:
    :param y_range:
    :param x_nodes:
    :param y_nodes:
    :return:
    """
    process = Process(target=_display_plot, args=(x_range, y_range))
    process.start()


def _display_plot(x_range, y_range):
    """
        Display plot of interpolated values using pyplot
    :param x_range: range of arguments, had interpolated
    :param y_range: range of interpolated values
    :param x_nodes: interpolation nodes args
    :param y_nodes: interpolation nodes values
    """
    plt.plot(x_range, y_range, label='interpolation')  # interpolated function

    plt.ioff()              # disable window interactive mode
    plt.legend(loc='best')  # enable labels in plot
    plt.show()


def main():
    """ main logic """
    def func(x):
        return (2 * math.cos(x) + 3 * math.sin(x)) / ((2 * math.sin(x) - 3 * math.cos(x)) ** 3)

    start, end = 1.0, 2.0

    x_range = np.arange(start, end, 0.0001)
    y_range = np.array([func(x) for x in x_range])
    _display_plot_async(x_range, y_range)

    integr_3 = chebyshev_integration(func, start, end, n=3)
    integr_4 = chebyshev_integration(func, start, end, n=4)
    print('Chebyshev integration\nn = 3: {}\nn = 4: {}'.format(integr_3, integr_4))

    print('\nGeneric methods')
    simps = simpson_integration(func, start, end, step=0.001)
    trapez = trapeze_integration(func, start, end, step=0.001)
    print('trapeze: {}, simpson: {}'.format(trapez, simps))


if __name__ == '__main__':
    main()
