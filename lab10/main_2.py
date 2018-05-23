"""
    Numeric analysis methods
    Laboratory work number 10
    Variant 8, Zaharov Igor

    Numerical integration, Task 2, execution logic
"""

import math
import numpy as np
from main_1 import trapeze_integration, simpson_integration
from chebyshev import chebyshev_integration
from plotting import display_plot_async


def main():
    """ main logic """
    def func1(x):
        # TODO: solve problem with this function
        return (2 * math.cos(x) + 3 * math.sin(x)) / ((2 * math.sin(x) - 3 * math.cos(x)) ** 3)

    def func(x):
        # return (math.cos(x) - x) * math.exp(x ** 2)
        # return math.sin(x) * math.exp(x ** 2)
        return (x + math.cos(x)) / (x**2 + 2 * math.sin(x))

    # problematic function borders
    # start, end = 1, 2

    # start, end = -1.7, 0
    # start, end = 0.7, 1.7
    start, end = math.pi, 2 * math.pi

    x_range = np.arange(start, end, 0.0001)
    y_range = np.array([func(x) for x in x_range])
    display_plot_async(x_range, y_range)

    integr_3 = chebyshev_integration(func, start, end, n=3)
    integr_4 = chebyshev_integration(func, start, end, n=4)
    print('Chebyshev integration\nn = 3: {}\nn = 4: {}'.format(integr_3, integr_4))

    print('\nGeneric methods')
    simps = simpson_integration(func, start, end, step=0.001)
    trapez = trapeze_integration(func, start, end, step=0.001)
    print('trapeze: {}, simpson: {}'.format(trapez, simps))


if __name__ == '__main__':
    main()
