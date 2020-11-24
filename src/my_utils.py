import torch as T
import numpy as np
import math as m
import yaml
import src.policies as policies
from opensimplex import OpenSimplex
import time

class SimplexNoise:
    """
    A simplex action noise
    """
    def __init__(self, dim, scale):
        super().__init__()
        self.idx = 0
        self.dim = dim
        self.scale = scale
        self.noisefun = OpenSimplex(seed=int((time.time() % 1) * 10000000))

    def __call__(self) -> np.ndarray:
        self.idx += 1
        return np.array([(self.noisefun.noise2d(x=self.idx / float(self.scale), y=i*10.) + self.noisefun.noise2d(x=self.idx / (self.scale * 7.), y=i*10.)) for i in range(self.dim)])

    def __repr__(self) -> str:
        return 'Opensimplex Noise()'.format()

def to_tensor(x, add_batchdim=False):
    x = T.FloatTensor(x.astype(np.float32))
    if add_batchdim:
        x = x.unsqueeze(0)
    return x

def import_env(name):
    if name == "hexapod_wp":
        from src.envs.bullet_hexapod.hexapod_wp import HexapodBulletEnv as env_fun
    if name == "hexapod_straight":
        from src.envs.bullet_hexapod.hexapod_straight import HexapodBulletEnv as env_fun
    elif name == "quadrotor_stab":
        from src.envs.bullet_quadrotor.quadrotor_stab import QuadrotorBulletEnv as env_fun
    elif name == "quadrotor_vel":
        from src.envs.bullet_quadrotor.quadrotor_vel import QuadrotorBulletEnv as env_fun
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

def _quat_to_euler(x, y, z, w):
        pitch =  -m.asin(2.0 * (x*z - w*y))
        roll  =  m.atan2(2.0 * (w*x + y*z), w*w - x*x - y*y + z*z)
        yaw   =  m.atan2(2.0 * (w*z + x*y), w*w + x*x - y*y - z*z)
        return [roll, pitch, yaw]

def _euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
        yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
        yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)
    return [qx, qy, qz, qw]

if __name__=="__main__":
    noise = SimplexNoise(4, 5)
    for i in range(1000):
        print(noise())