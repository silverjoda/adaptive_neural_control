import torch as T
import numpy as np
import math
import src.policies as policies

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
