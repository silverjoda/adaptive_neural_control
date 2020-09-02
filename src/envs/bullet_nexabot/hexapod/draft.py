import numpy as np
import matplotlib.pyplot as plt
from opensimplex import OpenSimplex
import time
from abc import ABC, abstractmethod
from stable_baselines.common.noise import ActionNoise

class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class SimplexNoise(ActionNoise):
    """
    A simplex action noise
    """

    def __init__(self):
        super().__init__()
        self.idx = 0
        self.noisefun = OpenSimplex(seed=int((time.time() % 1) * 10000000))

    def __call__(self) -> np.ndarray:
        self.idx += 1
        return self.noisefun.noise2d(x=self.idx / 10, y=0) + self.noisefun.noise2d(x=self.idx / 50, y=0)

    def __repr__(self) -> str:
        return 'Opensimplex Noise()'.format()

#ou = OrnsteinUhlenbeckActionNoise(mu=np.array([0]), sigma=0.2, theta=0.15, dt=1e-1, x0=None)
#noise = [ou() for i in range(100)]

#noisefun = OpenSimplex(seed=int((time.time() % 1) * 10000000))
#noise = [noisefun.noise2d(x=i/10, y=0) + noisefun.noise2d(x=i/50, y=0) for i in range(100)]

noisefun = SimplexNoise()
noise = [noisefun() for _ in range(100)]

plt.plot(range(100), noise)
plt.show()

