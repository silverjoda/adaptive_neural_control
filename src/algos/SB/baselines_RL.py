from stable_baselines import PPO2, A2C, TD3, SAC
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines.common.policies import LstmPolicy
import src.my_utils as my_utils
import time
import random
import string
import socket
import argparse
import yaml
import os
from pprint import pprint
from shutil import copyfile
import numpy as np
from copy import deepcopy

class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[128, 'lstm', dict(vf=[128, 64], pi=[128])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description='Pass in parameters. ')
    parser.add_argument('--train',  action='store_true', required=False,
                        help='Flag indicating whether the training process is to be run. ')
    parser.add_argument('--test', action='store_true', required=False,
                        help='Flag indicating whether the testing process is to be run. ')
    parser.add_argument('--animate', action='store_true', required=False,
                        help='Flag indicating whether the environment will be rendered. ')
    parser.add_argument('--test_agent_path', type=str, default=".", required=False,
                        help='Path of test agent. ')
    parser.add_argument('--algo_config', type=str, default="configs/td3_default_config.yaml", required=False,
                        help='Algorithm config file name. ')
    parser.add_argument('--env_config', type=str, required=True,
                        help='Env config file name. ')
    parser.add_argument('--iters', type=int, required=False, default=200000, help='Number of training steps. ')

    args = parser.parse_args()
    return args.__dict__

def read_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def make_model(config, env, action_noise_fun):
    model = None

    if config["algo_name"] == "A2C" and config["policy_name"] == "MlpPolicy":
        model = A2C(config["policy_name"],
            env=env,
            gamma=config["gamma"],
            n_steps=config["n_steps"],
            vf_coef=config["vf_coef"],
            ent_coef = config["ent_coef"],
            max_grad_norm=config["max_grad_norm"],
            learning_rate=config["sb2_learning_rate"],
            alpha=config["alpha"],
            epsilon=config["epsilon"],
            lr_schedule=config["lr_schedule"],
            verbose=config["verbose"],
            tensorboard_log="./tb/{}/".format(config["session_ID"]),
            full_tensorboard_log=config["full_tensorboard_log"],
            policy_kwargs=dict(net_arch=[int(config["policy_hid_dim"]), int(config["policy_hid_dim"])]))

    if config["algo_name"] == "A2C" and config["policy_name"] == "MlpLstmPolicy":
        model = A2C(CustomLSTMPolicy,
            env=env,
            gamma=config["gamma"],
            n_steps=config["n_steps"],
            vf_coef=config["vf_coef"],
            ent_coef = config["ent_coef"],
            max_grad_norm=config["max_grad_norm"],
            learning_rate=config["sb2_learning_rate"],
            alpha=config["alpha"],
            epsilon=config["epsilon"],
            lr_schedule=config["lr_schedule"],
            verbose=config["verbose"],
            tensorboard_log="./tb/{}/".format(config["session_ID"]),
            full_tensorboard_log=config["full_tensorboard_log"])

    assert model is not None, "Alg name not found, exiting. "
    return model

def load_model(config):
    model = None
    if config["algo_name"] == "TD3":
        model = TD3.load("agents/{}".format(args["test_agent_path"]))
    if config["algo_name"] == "A2C":
        model = A2C.load("agents/{}".format(args["test_agent_path"]))
    if config["algo_name"] == "SAC":
        model = SAC.load("agents/{}".format(args["test_agent_path"]))
    if config["algo_name"] == "PPO2":
        model = PPO2.load("agents/{}".format(args["test_agent_path"]))
    assert model is not None, "Alg name not found, cannot load model, exiting. "
    return model

def test_agent(env, model, deterministic=True, N=100, print_rew=True):
    total_rew = 0
    for _ in range(N):
        obs = env.reset()
        episode_rew = 0
        while True:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            if hasattr(env, "get_original_reward"):
                reward = env.get_original_reward()
            episode_rew += reward
            total_rew += reward
            if done:
                if print_rew:
                    print(episode_rew)
                break
    return total_rew

def test_agent_rnn(env, model, deterministic=True, N=100, print_rew=True):
    total_rew = 0
    for _ in range(N):
        obs = env.reset()
        episode_rew = 0
        while True:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            if hasattr(env, "get_original_reward"):
                reward = env.get_original_reward()
            episode_rew += reward[0]
            total_rew += reward[0]
            # env.render()
            if done[0]:  # .all() for rnn
                if print_rew:
                    print(episode_rew)
                break
    return total_rew


def setup_train(config, setup_dirs=True):
    if setup_dirs:
        for s in ["agents", "agents_cp", "tb"]:
            if not os.path.exists(s):
                os.makedirs(s)

    # Random ID of this session
    if config["default_session_ID"] is None:
        config["session_ID"] = ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3))
    else:
        config["session_ID"] = config["default_session_ID"]

    stats_path = "agents/{}_vecnorm.pkl".format(config["session_ID"])

    # Import correct env by name
    env_fun = my_utils.import_env(config["env_name"])
    if config["dummy_vec_env"]:
        vec_env = DummyVecEnv([lambda : env_fun(config) for _ in range(config["n_envs"])])
    else:
        vec_env = SubprocVecEnv([lambda : env_fun(config) for _ in range(config["n_envs"])], start_method='fork')
    env = VecNormalize(vec_env,
                       gamma=config["gamma"],
                       norm_obs=config["norm_obs"],
                       norm_reward=config["norm_reward"])
    model = make_model(config, env, None)

    checkpoint_callback = CheckpointCallback(save_freq=300000,
                                             save_path='agents_cp/',
                                             name_prefix=config["session_ID"], verbose=1)


    return env, model, checkpoint_callback, stats_path

def setup_eval(config, stats_path, seed=0):
    env_fun = my_utils.import_env(config["env_name"])
    config_tmp = deepcopy(config)
    config_tmp["seed"] = seed
    env = DummyVecEnv([lambda : env_fun(config) for _ in range(config["n_envs"])])
    return env

if __name__ == "__main__":
    args = parse_args()
    algo_config = read_config(args["algo_config"])
    env_config = read_config(args["env_config"])
    config = {**args, **algo_config, **env_config}

    if args["train"] or socket.gethostname() == "goedel":
        env, model, checkpoint_callback, stats_path = setup_train(config)

        t1 = time.time()
        model.learn(total_timesteps=algo_config["iters"], callback=checkpoint_callback)
        t2 = time.time()

        # Make tb run script inside tb dir
        if os.path.exists(os.path.join("tb", config["session_ID"])):
            copyfile("tb_runner.py", os.path.join("tb", config["session_ID"], "tb_runner.py"))

        print("Training time: {}".format(t2-t1))
        pprint(config)

        model.save("agents/{}_SB_policy".format(config["session_ID"]))
        env.save(stats_path)
        env.close()

    if args["test"] and socket.gethostname() != "goedel":
        stats_path = "agents/{}_vecnorm.pkl".format(args["test_agent_path"][:3])
        env_fun = my_utils.import_env(env_config["env_name"])
        #env = env_fun(config)  # Default, without normalization
        env = DummyVecEnv([lambda: env_fun(config)])
        #env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False

        model = load_model(config)

        test_agent(env, model, deterministic=True)