"""
    Chebyshev integration method
"""


t_n = {
    3: [0.707107, 0, -0.707107],
    4: [0.794654, 0.187592, -0.187592, -0.794654],
}

c_n = {
    3: 2.0 / 3,
    4: 0.5
}


def chebyshev_integration(func, start: float, end: float, n: int):
    """

    :param func:
    :param start:
    :param end:
    :param n:
    :return:
    """
    integr_sum = 0
    for coef in t_n[n]:
        x = (end + start) / 2 + ((end - start) / 2) * coef
        integr_sum += func(x)

    integr_sum *= (end - start) / n
    return integr_sum
