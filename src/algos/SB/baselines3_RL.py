from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
import src.my_utils as my_utils
import time
import random
import socket
import argparse
import yaml
import os
from pprint import pprint
from shutil import copyfile
from custom_policies import customActorCriticPolicyWrapper
from copy import deepcopy
import numpy as np


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        pass

    def _on_rollout_end(self) -> None:
        # Log scalar value (here a random variable)
        max_returns = np.max(self.locals['rollout_buffer'].returns, axis=0).mean()
        self.logger.record('mean_max_returns', max_returns)

        return None

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

def make_model(config, env):
    policy = config["policy_name"]

    if config["policy_name"] == "CustomTCNPolicy":
        # TODO: Continue here
        policy = customActorCriticPolicyWrapper(env.)

    tb_log = None
    if config["tensorboard_log"]:
        tb_log = "./tb/{}/".format(config["session_ID"])

    model = A2C(policy=policy,
                env=env,
                gamma=config["gamma"],
                n_steps=config["n_steps"],
                vf_coef=config["vf_coef"],
                ent_coef = config["ent_coef"],
                max_grad_norm=config["max_grad_norm"],
                learning_rate=eval(config["learning_rate"]),
                verbose=config["verbose"],
                use_sde=config["use_sde"],
                tensorboard_log=tb_log,
                device="cpu",
                policy_kwargs=dict(net_arch=[int(config["policy_hid_dim"]), int(config["policy_hid_dim"])]))

    exit()

    return model

def test_agent(env, model, deterministic=True, N=100, print_rew=True, render=True):
    total_rew = 0
    for _ in range(N):
        obs = env.reset()
        episode_rew = 0
        while True:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            episode_rew += reward
            total_rew += reward
            if render:
                env.render()
            if done:
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
    env = VecNormalize(SubprocVecEnv([lambda : env_fun(config) for _ in range(config["n_envs"])], start_method='fork'),
                       gamma=config["gamma"],
                       norm_obs=config["norm_obs"],
                       norm_reward=config["norm_reward"])
    model = make_model(config, env)

    checkpoint_callback = CheckpointCallback(save_freq=300000,
                                             save_path='agents_cp/',
                                             name_prefix=config["session_ID"], verbose=1)

    log_rew_callback = TensorboardCallback(verbose=1)
    callback_list = CallbackList([checkpoint_callback, log_rew_callback])

    return env, model, callback_list, stats_path

def setup_eval(config, stats_path, seed=1337):
    env_fun = my_utils.import_env(config["env_name"])
    config_tmp = deepcopy(config)
    config_tmp["seed"] = seed
    env = VecNormalize(DummyVecEnv([lambda: env_fun(config_tmp)]),
                       gamma=config["gamma"],
                       norm_obs=config["norm_obs"],
                       norm_reward=config["norm_reward"])
    VecNormalize.load(stats_path, env)
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

        print("Training time: {}".format(t2 - t1))
        pprint(config)

        model.save("agents/{}_SB_policy".format(config["session_ID"]))
        env.save(stats_path)
        env.close()

    if args["test"] and socket.gethostname() != "goedel":
        stats_path = "agents/{}_vecnorm.pkl".format(args["test_agent_path"][:3])
        env_fun = my_utils.import_env(env_config["env_name"])
        # env = env_fun(config)  # Default, without normalization
        env = DummyVecEnv([lambda: env_fun(config)])
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False

        model = A2C.load("agents/{}".format(args["test_agent_path"]))
        test_agent(env, model, deterministic=True)
