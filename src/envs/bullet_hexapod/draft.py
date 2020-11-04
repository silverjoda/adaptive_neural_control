import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,1,100)
y = np.sin(t * np.pi * 0.5)
plt.plot(t,y)
plt.show()