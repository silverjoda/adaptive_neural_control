import torch as T
import numpy as np
import math
import yaml
import src.policies as policies
from opensimplex import OpenSimplex
import time



class SimplexNoise:
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
        return np.array([(self.noisefun.noise2d(x=self.idx / 2., y=i*10.) + self.noisefun.noise2d(x=self.idx / 10., y=i*10.)) for i in range(self.dim)])

    def __repr__(self) -> str:
        return 'Opensimplex Noise()'.format()

def to_tensor(x, add_batchdim=False):
    x = T.FloatTensor(x.astype(np.float32))
    if add_batchdim:
        x = x.unsqueeze(0)
    return x

def import_env(name):
    if name == "hexapod":
        from src.envs.bullet_hexapod.hexapod import HexapodBulletEnv as env_fun
    elif name == "quadrotor":
        from src.envs.bullet_quadrotor.quadrotor import QuadrotorBulletEnv as env_fun
    elif name == "buggy":
        from src.envs.bullet_buggy.buggy import BuggyBulletEnv as env_fun
    elif name == "quadruped":
        from src.envs.bullet_quadruped.quadruped import QuadrupedBulletEnv as env_fun
    elif name == "cartpole_balance":
        from src.envs.bullet_cartpole_balance.cartpole_balance import CartPoleBulletEnv as env_fun
    else:
        raise TypeError
    return env_fun

def make_policy(env, config):
    if config["policy_type"] == "slp":
        return policies.SLP_PG(env, config)
    elif config["policy_type"] == "mlp":
        return policies.NN_PG(env, config)
    elif config["policy_type"] == "mlp_old":
        return policies.NN_PG_OLD(env, config)
    elif config["policy_type"] == "mlp_def":
        return policies.NN_PG_DEF(env, config)
    elif config["policy_type"] == "rnn":
        return policies.RNN_PG(env, config)
    if config["policy_type"] == "cyc_quad":
        return policies.CYC_QUAD(env, config)
    if config["policy_type"] == "cyc_hex":
        return policies.CYC_HEX(env, config)
    else:
        raise TypeError

def read_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def make_action_noise_policy(env, config):
    return None

if __name__=="__main__":
    noise = SimplexNoise(3)
    for i in range(1000):
        print(noise())