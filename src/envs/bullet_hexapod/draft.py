import numpy as np

[print(i, i % (2 * np.pi), np.sin(i), np.sin(i), np.sin(i % (2 * np.pi))) for i in range(9999900,10000000)]

