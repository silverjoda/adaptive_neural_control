import numpy as np
import matplotlib.pyplot as plt

N = 1000
pts = np.random.rand(2, N) * 2 - 1
pts_normed = pts / np.sqrt(np.sum(np.power(pts, 2), axis=0))
pts_rnd_scaled = pts_normed * np.random.rand(1, N)
plt.plot(pts_rnd_scaled[0], pts_rnd_scaled[1], 'bo')
plt.show()
