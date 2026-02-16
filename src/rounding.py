from math import floor, isnan, log10


def significant_digits(x):
    return -int(floor(log10(abs(x))))


def significant_round(x, n=0):
    if x == 0:
        return 0.0

    if isnan(x):
        return x

    return round(x, significant_digits(x) + n)


def err_round(x):
    mean = x["mean"]
    std = x["std"]
    if std:
        n = significant_digits(std)
    else:
        n = 2
        std = 0
    return rf"{round(mean, n)} \pm {round(std, n)}"
