import numpy as np
import matplotlib.pyplot as plt

def f(x,a,c):
    coeff = np.abs(a - 2) / a
    inner_bracket = np.square(x / c) / np.abs(a - 2) + 1
    f = coeff * (np.power(inner_bracket, (a / 2)) - 1)
    return f

a = -1
c = 0.1
x = np.linspace(-5,5,500)
y = f(x, a, c)
plt.plot(x,y)
plt.show()
