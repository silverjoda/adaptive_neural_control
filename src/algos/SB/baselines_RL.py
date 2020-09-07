import gym
import sys
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, A2C, TD3
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
import time
import random
import string
import socket
import numpy as np
import argparse
import yaml
import os

def make_env(config, env_fun):
    def _init():
        env = env_fun(config)
        return env
    return _init

def parse_args():
    parser = argparse.ArgumentParser(description='Pass in parameters. ')
    parser.add_argument('--train', type=bool, default=False, required=False,
                        help='Flag indicating whether the training process is to be run. ')
    parser.add_argument('--test', type=bool, default=False, required=False,
                        help='Flag indicating whether the testing process is to be run. ')
    parser.add_argument('--test_agent_path', type=str, default=".", required=False,
                        help='Path of test agent. ')
    parser.add_argument('--animate', type=bool, default=False, required=False,
                        help='Flag indicating whether the environment will be rendered. ')
    parser.add_argument('--algo_config', type=str, default="td3_default_config.yaml", required=False,
                        help='Algorithm config flie name. ')
    parser.add_argument('--env_config', type=str, default="hexapod_config.yaml", required=False,
                        help='Env config file name. ')
    parser.add_argument('--iters', type=int, required=False, default=200000, help='Number of training steps. ')

    args = parser.parse_args()
    return args.__dict__

def read_config(path):
    with open(os.path.join('configs/', path)) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def import_env(name):
    env_fun = None
    if name == "hexapod":
        from src.envs.bullet_nexabot.hexapod.hexapod import HexapodBulletEnv as env_fun
    if name == "quadrotor":
        from src.envs.bullet_quadrotor.quadrotor import QuadrotorBulletEnv as env_fun
    if name == "buggy":
        from src.envs.bullet_buggy.buggy import BuggyBulletEnv as env_fun
    assert env_fun is not None, "Env name not found, exiting. "
    return env_fun

def make_model(config, env, action_noise_fun):
    model = None
    if config["algo_name"] == "TD3":
        model = TD3('MlpPolicy',
                    env=env,
                    gamma=config["gamma"],
                    learning_rate=config["learning_rate"],
                    buffer_size=config["buffer_size"],
                    learning_starts=config["learning_starts"],
                    train_freq=config["train_freq"],
                    gradient_steps=config["gradient_steps"],
                    batch_size=config["batch_size"],
                    tau=config["tau"],
                    policy_delay=config["policy_delay"],
                    action_noise=action_noise_fun,
                    target_policy_noise=config["target_policy_noise"],
                    target_noise_clip=config["target_noise_clip"],
                    verbose=config["verbose"],
                    tensorboard_log="./tb/{}/".format(session_ID),
                    policy_kwargs=config["policy_kwargs"])

    if config["algo_name"] == "A2C":
        A2C('MlpPolicy',
            env=env,
            gamma=algo_config["gamma"],
            n_steps=algo_config["n_steps"],
            vf_coef=algo_config["vf_coef"],
            ent_coef = algo_config["ent_coef"],
            max_grad_norm=algo_config["max_grad_norm"],
            learning_rate=algo_config["learning_rate"],
            alpha=algo_config["alpha"],
            epsilon=algo_config["epsilon"],
            lr_schedule=algo_config["lr_schedule"],
            verbose=algo_config["verbose"],
            tensorboard_log="./tb/{}/".format(session_ID),
            full_tensorboard_log=algo_config["full_tensorboard_log"],
            policy_kwargs=algo_config["policy_kwargs"])

    assert model is not None, "Alg name not found, exiting. "
    return model

def make_action_noise_fun(config):
    return None

def test_agent(env, model, deterministic=True):
    for _ in range(100):
        obs = env.reset()
        cum_rew = 0
        while True:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            cum_rew += reward
            env.render()
            if done:
                print(cum_rew)
                break
    env.close()

if __name__ == "__main__":
    args = parse_args()
    algo_config = read_config(args["algo_config"])
    env_config = read_config(args["env_config"])
    aug_env_config = {**args, **env_config}

    print(args)
    print(algo_config)
    print(env_config)

    # Import correct env by name
    env_fun = import_env(env_config["env_name"])

    # Random ID of this session
    session_ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))

    if args["train"] or socket.gethostname() == "goedel":
        n_envs = 10 if socket.gethostname() == "goedel" else 1

        env = SubprocVecEnv([make_env(aug_env_config, env_fun) for _ in range(n_envs)], start_method='fork')
        model = make_model(algo_config, env, None)

        checkpoint_callback = CheckpointCallback(save_freq=50000,
                                                 save_path='agents_cp/',
                                                 name_prefix=session_ID, verbose=1)

        t1 = time.time()
        model.learn(total_timesteps=int(algo_config["iters"]), callback=checkpoint_callback)
        t2 = time.time()

        print("Training time: {}".format(t2-t1))
        print(args)
        print(algo_config)
        print(env_config)

        model.save("agents/{}_SB_policy".format(session_ID))
        env.close()

    if args["test"] and socket.gethostname() != "goedel":
        env = make_env(aug_env_config, env_fun)

        if not args["train"]:
            model = A2C.load("agents/{}".format(args["test_agent_path"]))

        test_agent(env, model, deterministic=True)