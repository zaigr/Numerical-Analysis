"""
    This class represents 2-nd system from the given task
    1) sin(x + y) - 1.5 * x - 0.1 = 0
    2) x^2 + y^2 - 1 = 0
"""

import math
import sympy


class System2:
    """ class represent second system given from the task """
    def f1(self, x1, x2):
        """ first function of the system """
        return math.sin(x1 + x2) - 1.5 * x1 - 0.1

    def f2(self, x1, x2):
        """ second function of the system """
        return x1 ** 2 + x2 ** 2 - 1

    def df1dx1(self, x1, x2):
        """ partial derivative by x1 of the first system """
        return math.cos(x1 + x2) - 1.5

    def df1dx2(self, x1, x2):
        """ partial derivative by x2 of the first system """
        return math.cos(x1 + x2)

    def df2dx1(self, x1, x2):
        """ partial derivative by x1 of the second system """
        return 2 * x1

    def df2dx2(self, x1, x2):
        """ partial derivative by x2 of the second system """
        return 2 * x2

    def display_plot(self, start, end):
        """ display implicit plot of system using sympy.plotting """
        # configuring implicit sympy.plotting plot
        x, y = sympy.symbols('x y')
        # plot of the first system function
        p1 = sympy.plotting.plot_implicit(sympy.sin(x + y) - 1.5 * x - 0.1, (x, start, end),
                                          title='system 2', depth=4, points=800, show=False, adaptive=False)
        # plot of the second system function
        p2 = sympy.plotting.plot_implicit(sympy.Eq(x**2 + y**2, 1),
                                          title='system 2', depth=5, show=False, adaptive=False)
        # concat two implicit plots and show
        p1.extend(p2)
        p1.show()
