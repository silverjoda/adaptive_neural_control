import numpy as np

a = np.array([[-5, -3, 1.44]])
b = np.array([[-5, 1, 4.44]])

c = (a - a.mean()).T * (b - b.mean())

print(c)
