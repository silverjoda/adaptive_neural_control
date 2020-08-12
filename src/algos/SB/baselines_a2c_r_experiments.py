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
from copy import deepcopy

def make_env(params):
    def _init():
        env = env_fun(animate=params["animate"],
                      max_steps=params["max_steps"],
                      action_input=True,
                      latent_input=False,
                      is_variable=params["is_variable"])
        return env
    return _init

if __name__ == "__main__":
    args = ["None", "perlin", "straight_rough"]
    if len(sys.argv) > 1:
        args = sys.argv

    from src.envs.bullet_cartpole.hangpole_goal_cont_variable.hangpole_goal_cont_variable import \
        HangPoleGoalContVariableBulletEnv as env_fun

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"iters": 50000000,
              "batchsize": 60,
              "max_steps": 200,
              "gamma": 0.99,
              "policy_lr": 0.0001,
              "normalize_rewards": False,
              "animate": False,
              "is_variable": True,
              "train": True,
              "terrain": args[1],
              "r_type": args[2],
              "note": "Training: {}, {}, |Training without contacts| ".format(args[1], args[2]),
              "ID": ID}

    print(params)
    TRAIN = False
    n_envs = 8

    if TRAIN or socket.gethostname() == "goedel":
        if socket.gethostname() == "goedel": n_envs = 8
        env = SubprocVecEnv([make_env(params) for _ in range(n_envs)], start_method='fork')
        policy_kwargs = dict()

        model = A2C('MlpLstmPolicy',
                    env=env,
                    learning_rate=params["policy_lr"],
                    verbose=1,
                    n_steps=60,
                    ent_coef=0.0,
                    vf_coef=0.5,
                    lr_schedule='linear',
                    tensorboard_log="./tb/{}/".format(ID),
                    full_tensorboard_log=False,
                    gamma=params["gamma"],
                    policy_kwargs=policy_kwargs)

        # Save a checkpoint every 1000000 steps
        checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='agents_cp/',
                                                 name_prefix=params["ID"], verbose=1)

        # Train the agent
        print("Started training")
        t1 = time.time()
        model.learn(total_timesteps=int(params["iters"]), callback=checkpoint_callback)
        t2 = time.time()
        print("Training time: {}".format(t2-t1))
        print(params)
        model.save("agents/{}_SB_policy".format(params["ID"]))
        env.close()

    if socket.gethostname() == "goedel":
        exit()

    env_list = []
    for i in range(n_envs):
        params_tmp = deepcopy(params)
        if i == 0:
            params_tmp["animate"] = True
        else:
            params_tmp["animate"] = False
        env_list.append(make_env(params_tmp))
    env = SubprocVecEnv(env_list, start_method='fork')

    if not TRAIN:
        model = A2C.load("agents/826_SB_policy.zip")
        #model = A2C.load("agents_cp/42T_29100000_steps.zip")

    obs = env.reset()
    for _ in range(100):
        cum_rew = 0

        for i in range(100):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            cum_rew += reward[0]
            time.sleep(0.01)
            if done[0]:
                obs = env.reset()
                print(cum_rew)
                break

    env.close()