"""
    Numeric analysis methods
    Laboratory work number 10
    Variant 8, Zaharov Igor

    Numerical integration, Task 1, execution logic
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process


def simpson_integration(func, start, end, parts=2, step=None):
    """

    :param func:
    :param start:
    :param end:
    :return:
    """
    # check input
    if start >= end or not _is_even(parts):
        raise ValueError('Invalid segment!')

    step = (end - start) / parts if step is None else step
    x_n = np.arange(start, end, step)

    x_odd = [x_n[i] for i in range(len(x_n)) if not _is_even(i)]
    x_even = [x_n[i] for i in range(len(x_n)) if _is_even(i)]

    integr_sum = 0
    for i in range(len(x_odd)):
        integr_sum += 2 * func(x_even[i]) + 4 * func(x_odd[i])
    integr_sum *= step / 3

    return integr_sum


def _is_even(numb):
    return numb % 2 == 0


def trapeze_integration(func, start, end, parts=3, step=None):
    """

    :param func:
    :param start:
    :param end:
    :param parts:
    :return:
    """
    # check input
    if start >= end:
        raise ValueError('Invalid segment borders!')

    if step is None:
        step = (end - start) / parts
    else:
        parts = int((end - start) / step)
    integr_sum = (func(start) + func(end)) / 2
    x = np.arange(start, end, step)

    for i in range(1, len(x) - 1):
        integr_sum += func(x[i])

    return integr_sum * step


def rectangle_integration(func, start, end, parts=3, method='l', step=None):
    """

    :param func: function, should be integrated
    :param start: start of the segment
    :param end: end of the segment
    :param parts: number of splitting parts
    :param method: type of integration methods, left - 'l' and right - 'r'
    :param step:
    :return: approximate integral value
    """
    # check input
    if start >= end:
        raise ValueError('Invalid segment borders!')

    if step is None:
        step = (end - start) / parts
    else:
        parts = int((end - start) / step)

    integr_sum = 0
    x = start
    for i in range(parts - 1):
        if method == 'l':
            integr_sum += func(x) * step
            x += step
        elif method == 'r':
            x += step
            integr_sum += func(x) * step

    return integr_sum


def make_polynomial(coefs):
    """

    :param coefs:
    :return:
    """
    def func(x):
        value = 0
        for i in range(len(coefs)):
            value += coefs[i] * (x ** i)
        return value

    return func


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
    with open('points.json', 'r') as file:
        data = json.load(file)
        c = data['c']
        start = data['start']
        end = data['end']

    func = make_polynomial(c)

    print('simple methods')
    rectang = rectangle_integration(func, start, end)
    trapez = trapeze_integration(func, start, end)
    simps = simpson_integration(func, start, end)
    print('rectangle: {}, trapeze: {}, simpson: {}'.format(rectang, trapez, simps))

    step = 0.1
    print('\ngeneric methods with step {}'.format(step))
    rectang = rectangle_integration(func, start, end, step=step)
    trapez = trapeze_integration(func, start, end, step=step)
    simps = simpson_integration(func, start, end, step=step)
    print('rectangle: {}, trapeze: {}, simpson: {}'.format(rectang, trapez, simps))

    x_range = np.arange(start, end, 0.001)
    y_range = np.array([func(x) for x in x_range])
    _display_plot_async(x_range, y_range)


if __name__ == '__main__':
    main()
