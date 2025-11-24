
def const(x):
    return lambda _: x


def fst(x):
    return x[0]


head = fst


def tail(x):
    return x[1:]


def snd(x):
    return x[1]


def validcheck(f=None):
    if f:
        return lambda x: f(x) is not None
    else:
        return lambda x: x is not None
