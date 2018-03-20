"""
    this class represents first system from given task
    1) 2 * y - cos(x + 1) = 0
    2) x + sin(y) + 0.4 = 0
"""

import math
import numpy as np
import matplotlib.pyplot as plt


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
        return math.asin(- x - 0.4)

    def display_plot(self, start, end, step):
        """ display implicit plot of system using pyplot """
        x1_range = np.arange(start, end, step)
        y1_range = np.array([self.expl_f1(x) for x in x1_range])  # first function point range

        f2_domain = -1.4, 0.6
        x2_range = np.arange(*f2_domain, step)
        y2_range = np.array([self.expl_f2(x) for x in x2_range])  # second function point range

        # configure system plot
        plt.title('system 1')
        plt.plot(x1_range, y1_range, color='red', linewidth=0.8, linestyle='--')
        plt.plot(x2_range, y2_range, color='blue', linewidth=1)

        # disable interactive mod and show
        plt.ioff()
        plt.show()
