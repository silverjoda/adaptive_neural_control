import gym
import sys
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, A2C
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
import time
import random
import string
import socket
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np
from copy import deepcopy

def make_env(params):
    def _init():
        env = env_fun(animate=params["animate"],
                      max_steps=params["max_steps"],
                      step_counter=True,
                      terrain_name=params["terrain"],
                      training_mode=params["r_type"],
                      variable_velocity=params["variable_velocity"])
        return env
    return _init

class PyTorchMlp(nn.Module):

    def __init__(self, n_inputs=30, n_actions=18):
        nn.Module.__init__(self)

        self.fc1 = nn.Linear(n_inputs, 96)
        self.fc2 = nn.Linear(96, 96)
        self.fc3 = nn.Linear(96, n_actions)
        self.activ_fn = nn.Tanh()
        self.out_activ = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.activ_fn(self.fc1(x))
        x = self.activ_fn(self.fc2(x))
        x = self.fc3(x)
        return x


def learn_model(params, env, policy, regressor):

    # Save trained model
    sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "agents/{}.zip".format(env.__class__.__name__, policy.__class__.__name__, params["ID"]))
    T.save(policy.state_dict(), sdir)
    print("Saved checkpoint at {} with params {}".format(sdir, params))

if __name__ == "__main__":
    from src.envs.bullet_nexabot.hexapod.hexapod_wip import HexapodBulletEnv as env_fun

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"episodes": 100,
              "batchsize": 60,
              "max_steps": 100,
              "gamma": 0.99,
              "policy_lr": 0.001,
              "weight_decay": 0.0001,
              "normalize_rewards": False,
              "animate": False,
              "variable_velocity": False,
              "train": True,
              "note": "",
              "ID": ID}

    print(params)
    TRAIN = False
    LOAD_POLICY = False

    if TRAIN or socket.gethostname() == "goedel":
        env = env_fun(animate=True,
                      max_steps=params["max_steps"],
                      step_counter=True,
                      terrain_name=params["terrain"],
                      training_mode=params["r_type"],
                      variable_velocity=False)

        # Here random or loaded learned policy
        policy = A2C('MlpPolicy', env)
        if LOAD_POLICY:
            policy_dir = "agents/xxx.zip"
            policy = A2C.load(policy_dir)  # 2Q5

        # Make regressor NN agent
        regressor = PyTorchMlp(env.obs_dim + env.act_dim, env.act_dim)

        # Train the agent
        t1 = time.time()
        learn_model(params, env, policy, regressor)
        t2 = time.time()
        print("Training time: {}".format(t2-t1))
        print(params)
        env.close()
