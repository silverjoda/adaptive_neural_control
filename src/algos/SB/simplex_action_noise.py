from stable_baselines.common.noise import ActionNoise
from opensimplex import OpenSimplex
import time
import numpy as np

class SimplexNoise(ActionNoise):
    """
    A simplex action noise
    """
    def __init__(self, dim):
        super().__init__()
        self.idx = 0
        self.dim = dim
        self.noisefun = OpenSimplex(seed=int((time.time() % 1) * 10000000))

    def __call__(self) -> np.ndarray:
        self.idx += 1
        return np.array([(self.noisefun.noise2d(x=self.idx / 10, y=i*10) + self.noisefun.noise2d(x=self.idx / 50, y=i*10)) for i in range(self.dim)])

    def __repr__(self) -> str:
        return 'Opensimplex Noise()'.format()