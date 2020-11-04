import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,1,100)
y1 = 1 - np.square(t - 1)
y2 = np.sin(t * np.pi * 0.5)
plt.plot(t, y1, t, y2, t, t)
plt.show()