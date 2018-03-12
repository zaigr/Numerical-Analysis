"""
    Numeric analysis methods
    Laboratory work number 4-5
    Variant 8, Zaharov Igor
"""

import math
import sympy
import scipy.misc
import numpy.linalg
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process


class System1:
    """ class represent first system given from the task """
    def f1(self, x1, x2):
        """ first function of the system """
        return 2 * x2 - math.cos(x1 + 1)

    def f2(self, x1, x2):
        """ second function of the system """
        return x1 + math.sin(x2) + 0.4

    def df1dx1(self, x1, x2):
        """ partial derivative by x1 of the first function """
        return math.sin(x1)

    def df1dx2(self, x1, x2):
        """ partial derivative by x2 of the first function """
        return 2

    def df2dx1(self, x1, x2):
        """ partial derivative by x1 of the second function """
        return 1

    def df2dx2(self, x1, x2):
        return math.cos(x2)

    def expl_f1(self, x):
        """ explicit first function of the system """
        return 1. / 2 * math.cos(x + 1)

    def expl_f2(self, x):
        """ explicit second function of the system """
        return - math.sin(x) - 0.4

    def display_plot(self, start, end, step):
        """ display implicit plot of system using pyplot """
        x_range = np.arange(start, end, step)
        y1_range = np.array([self.expl_f1(x) for x in x_range])  # first function point range
        y2_range = np.array([self.expl_f2(x) for x in x_range])  # second function point range

        # configure system plot
        plt.plot(x_range, y1_range, color='red', linewidth=0.8, linestyle='--')
        plt.plot(x_range, y2_range, color='blue', linewidth=1)

        # disable interactive mod and show
        plt.ioff()
        plt.show()


class System2:
    """ class represent second system given from the task """
    def f1(self, x1, x2):
        """ first function of the system """
        return math.sin(x1 + x2) - 1.5 * x1 - 0.1

    def f2(self, x1, x2):
        """ second function of the system """
        return x1 ** 2 + x2 ** 2 - 1

    def display_plot(self, start, end):
        """ display implicit plot of system using sympy.plotting """
        # configuring implicit sympy.plotting plot
        x, y = sympy.symbols('x y')
        # plot of the first system function
        p1 = sympy.plotting.plot_implicit(sympy.sin(x + y) - 1.5 * x - 0.1, (x, start, end),
                                          title='system 1', depth=4, show=False)
        # plot of the second system function
        p2 = sympy.plotting.plot_implicit(sympy.Eq(x**2 + y**2, 1),
                                          title='system 2', depth=5, show=False)

        # concat two implicit plots and show
        p1.extend(p2)
        p1.show()


def main():
    """ main logic """
    sys1 = System1()
    # plot compact borders
    l, r, step = -17, 17, 0.001
    plt1_window = Process(target=sys1.display_plot, args=(l, r, step))
    plt1_window.start()

    sys2 = System2()
    plt2_window = Process(target=sys2.display_plot, args=(l, r))
    plt2_window.start()

if __name__ == '__main__':
    main()
