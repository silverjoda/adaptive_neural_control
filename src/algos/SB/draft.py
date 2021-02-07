import optuna
from baselines3_RL import *
from copy import deepcopy
from multiprocessing import Pool, Process, Manager
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0,4 * np.pi, 300)
y = np.sin(t)
plt.plot(t,y)
plt.show()



