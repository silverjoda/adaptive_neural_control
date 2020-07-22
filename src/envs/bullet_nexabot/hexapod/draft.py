import numpy as np
import matplotlib.pyplot as plt
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

ou = OrnsteinUhlenbeckActionNoise(mu=np.array([0]), sigma=0.5, theta=0.15, dt=1e-1, x0=None)
noise = [ou() for i in range(100)]
plt.plot(range(100),noise)
plt.show()

