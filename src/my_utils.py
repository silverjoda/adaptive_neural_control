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
    if name == "hexapod":
        from src.envs.bullet_hexapod.hexapod import HexapodBulletEnv as env_fun
    elif name == "hexapod_wp":
        from src.envs.bullet_hexapod.hexapod_wp import HexapodBulletEnv as env_fun
    elif name == "hexapod_wp_eef":
        from src.envs.bullet_hexapod.hexapod_wp_eef import HexapodBulletEnv as env_fun
    elif name == "hexapod_wp_eef_direct":
        from src.envs.bullet_hexapod.hexapod_wp_eef_direct import HexapodBulletEnv as env_fun
    elif name == "hexapod_wp_eef_es":
        from src.envs.bullet_hexapod.hexapod_wp_eef_es import HexapodBulletEnv as env_fun
    elif name == "hexapod_wp_joint_phases":
        from src.envs.bullet_hexapod.hexapod_wp_joint_phases import HexapodBulletEnv as env_fun
    elif name == "hexapod_wp_joint_phases_es":
        from src.envs.bullet_hexapod.hexapod_wp_joint_phases_es import HexapodBulletEnv as env_fun
    elif name == "hexapod_straight":
        from src.envs.bullet_hexapod.hexapod_straight import HexapodBulletEnv as env_fun
    elif name == "hexapod_obstacle":
        from src.envs.bullet_hexapod.hexapod_obstacle import HexapodBulletEnv as env_fun
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
        return policies.SLP_PG(env.obs_dim, env.act_dim, config)
    elif config["policy_type"] == "mlp":
        return policies.PI_AC(env.obs_dim, env.act_dim, config)
    elif config["policy_type"] == "slp_es":
        return policies.SLP_ES(env.obs_dim, env.act_dim, config)
    elif config["policy_type"] == "mlp_es":
        return policies.MLP_ES(env.obs_dim, env.act_dim, config)
    elif config["policy_type"] == "ff_hex_eef":
        return policies.FF_HEX_EEF(env.obs_dim, env.act_dim, config)
    elif config["policy_type"] == "ff_hex_eef_adv":
        return policies.FF_HEX_EEF_ADVANCED(env.obs_dim, env.act_dim, config)
    elif config["policy_type"] == "ff_hex_joint_phases":
        return policies.FF_HEX_JOINT_PHASES(env.obs_dim, env.act_dim, config)
    else:
        raise TypeError

def make_par_policy(sub_proc_env, config):
    if config["policy_type"] == "slp":
        return policies.SLP_PG(sub_proc_env.observation_space.shape[0], sub_proc_env.action_space.shape[0], config)
    elif config["policy_type"] == "mlp":
        return policies.PI_AC(sub_proc_env.observation_space.shape[0], sub_proc_env.action_space.shape[0], config)
    elif config["policy_type"] == "ff_hex_eef":
        return policies.FF_HEX_EEF(sub_proc_env.observation_space.shape[0], sub_proc_env.action_space.shape[0], config)
    else:
        raise TypeError

def make_vf(env, config):
    if config["policy_type"] == "mlp":
        return policies.VF_AC(env.obs_dim, config)
    else:
        raise TypeError

def make_par_vf(sub_proc_env, config):
    if config["policy_type"] == "mlp":
        return policies.VF_AC(sub_proc_env.observation_space.shape[0], config)
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