"""
    Numeric analysis methods
    Laboratory work number 4-5
    Variant 8, Zaharov Igor
"""

import math
import numpy.linalg
import numpy as np
from multiprocessing import Process

from system1 import System1
from system2 import System2


def newton_method(system, approx_x1, approx_x2, fault=0.001):
    """
        Newton method of solving system of two nonlinear equations
    :param system: class, represent system of two nonlinear equations
    :param approx_x: approximation of system root
    :param fault: погрешность
    :return: tuple of two roots (x1, x2)
    """
    x1, x2 = approx_x1, approx_x2
    while math.fabs(system.f1(x1, x2)) > fault and math.fabs(system.f2(x1, x2)) > fault:
        # matrix a1 and a2 used in Kramer method
        a1 = np.array([[system.f1(x1, x2), system.df1dx2(x1, x2)],
                      [system.f2(x1, x2), system.df2dx2(x1, x2)]])
        a2 = np.array([[system.df1dx1(x1, x2), system.f1(x1, x2)],
                      [system.df2dx1(x1, x2), system.f2(x1, x2)]])
        jacobian = np.array([[system.df1dx1(x1, x2), system.df2dx1(x1, x2)],
                            [system.df1dx2(x1, x2), system.df2dx2(x1, x2)]])

        x1 = x1 - numpy.linalg.det(a1) / numpy.linalg.det(jacobian)
        x2 = x2 - numpy.linalg.det(a2) / numpy.linalg.det(jacobian)

    return x1, x2


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

    # event loop, can be interrupt by KeyboardInterrupt Ctrl+C
    while True:
        system_solving_dialog(sys1, 'FIRST')
        system_solving_dialog(sys2, 'SECOND')


def system_solving_dialog(system, sys_number: str):
    """ user interaction logic """
    print('\nInput approximated values of {} system root'.format(sys_number))

    x1 = float(input('input x1 value near the first root: '))
    x2 = float(input('input x2 value near the second root: '))

    newton_method_res = newton_method(system, x1, x2)
    print('{} system\nNewton method results: {} {}'.format(sys_number, *newton_method_res))


if __name__ == '__main__':
    main()
