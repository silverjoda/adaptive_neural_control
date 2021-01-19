import numpy as np

def sinifier(f):
    def wrapper(x):
        v = f(x)
        return np.sin(v)
    return wrapper

@sinifier
def func(x):
    return int(x + 1)

def numgen(N):
    for i in range(N):
        yield i

N = 10
gen = numgen(N)
print([func(i) for i in gen])
