from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
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
import torch as T

# class TensorboardCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super(TensorboardCallback, self).__init__(verbose)
#
#     def _on_step(self) -> bool:
#         pass
#
#     def _on_rollout_end(self) -> None:
#         # Log scalar value (here a random variable)
#         max_returns = np.max(self.locals['rollout_buffer'].returns, axis=0).mean()
#         mean_advantages = self.locals['rollout_buffer'].advantages.mean()
#         self.logger.record('train/mean_max_returns', max_returns)
#         self.logger.record('train/mean_advantages', mean_advantages)
#
#         return None

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
        policy = customActorCriticPolicyWrapper(env.observation_space.shape[0] // config["obs_input"], config["obs_input"])

    tb_log = None
    if config["tensorboard_log"]:
        tb_log = "./tb/{}/".format(config["session_ID"])

    ou_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape[0]), sigma=config["ou_sigma"] *  np.ones(env.action_space.shape[0]), theta=config["ou_theta"], dt=config["ou_dt"], initial_noise=None)

    model = TD3(policy=policy,
                env=env,
                buffer_size=config["buffer_size"],
                learning_starts=config["learning_starts"],
                action_noise=ou_noise,
                target_policy_noise=config["target_policy_noise"],
                target_noise_clip=config["target_noise_clip"],
                gamma=config["gamma"],
                tau=config["tau"],
                learning_rate=eval(config["learning_rate"]),
                verbose=config["verbose"],
                tensorboard_log=tb_log,
                device="cpu",
                policy_kwargs=dict(net_arch=[int(config["policy_hid_dim"]), int(config["policy_hid_dim"])]))

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

            if done:
                if print_rew:
                    print(episode_rew)
                break
    return total_rew

def test_agent_mirrored(env, model, deterministic=True, N=100, print_rew=True, render=True, perm=[0,1,2,3]):
    total_rew = 0

    for _ in range(N):
        # torso_quat, torso_vel, [signed_deviation], time_feature, scaled_joint_angles
        obs = env.reset()

        episode_rew = 0
        while True:
            torso_quat = obs[:4]
            torso_vel = obs[4:7]
            signed_deviation = obs[7]
            time_feature = obs[8]
            scaled_joint_angles = obs[9:]
            obs = [*[np.sign(perm[i]) * torso_quat[np.abs(perm[i])] for i in range(4)],
                   torso_vel[0], -torso_vel[1], torso_vel[2],
                   -signed_deviation,
                   time_feature,
                   *scaled_joint_angles[3:6], *scaled_joint_angles[0:3],
                    *scaled_joint_angles[9:12], *scaled_joint_angles[6:9],
                    *scaled_joint_angles[15:18], *scaled_joint_angles[12:15]]

            action, _states = model.predict(obs, deterministic=deterministic)
            action_mirrored = [*action[3:6], *action[0:3],
                    *action[9:12], *action[6:9],
                    *action[15:18], *action[12:15]]

            obs, reward, done, info = env.step(action_mirrored)
            episode_rew += reward
            total_rew += reward

            if done:
                if print_rew:
                    print(episode_rew)
                break
    return total_rew

def setup_train(config, setup_dirs=True):
    T.set_num_threads(1)
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
    env = env_fun(config)
    model = make_model(config, env)

    checkpoint_callback = CheckpointCallback(save_freq=100000,
                                             save_path='agents_cp/',
                                             name_prefix=config["session_ID"], verbose=1)

    # Separate evaluation env
    config_eval = deepcopy(config)
    config_eval["animate"] = False
    eval_env = env_fun(config_eval)
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env,
                                 eval_freq=10000,
                                 deterministic=True, render=False)
    callback_list = CallbackList([checkpoint_callback, eval_callback])

    return env, model, callback_list, stats_path

def setup_train_partial(config, setup_dirs=True):
    T.set_num_threads(1)
    if setup_dirs:
        for s in ["agents", "agents_cp", "tb"]:
            if not os.path.exists(s):
                os.makedirs(s)

    # Random ID of this session
    if config["default_session_ID"] is None:
        config["session_ID"] = ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3))
    else:
        config["session_ID"] = config["default_session_ID"]

    # Import correct env by name
    env_fun = my_utils.import_env(config["env_name"])
    env = env_fun(config)

    return env

def setup_eval(config, stats_path, seed=0):
    T.set_num_threads(1)
    env_fun = my_utils.import_env(config["env_name"])
    config_tmp = deepcopy(config)
    config_tmp["seed"] = seed
    return env_fun(config_tmp)

if __name__ == "__main__":
    args = parse_args()
    algo_config = read_config(args["algo_config"])
    env_config = read_config(args["env_config"])
    config = {**args, **algo_config, **env_config}

    if args["train"] or socket.gethostname() == "goedel":
        env, model, checkpoint_callback, stats_path = setup_train(config)

        t1 = time.time()
        model.learn(total_timesteps=algo_config["iters"], callback=checkpoint_callback, log_interval=1)
        t2 = time.time()

        # Make tb run script inside tb dir
        if os.path.exists(os.path.join("tb", config["session_ID"])):
            copyfile("tb_runner.py", os.path.join("tb", config["session_ID"], "tb_runner.py"))

        print("Training time: {}".format(t2 - t1))
        pprint(config)

        model.save("agents/{}_SB_policy".format(config["session_ID"]))
        env.close()

    if args["test"] and socket.gethostname() != "goedel":
        env_fun = my_utils.import_env(config["env_name"])
        config["seed"] = 1337

        env = env_fun(config)
        env.training = False
        env.norm_reward = False

        model = TD3.load("agents/{}".format(args["test_agent_path"]))
        # Normal testing
        N_test = 50
        total_rew = test_agent(env, model, deterministic=False, N=N_test)
        #total_rew = test_agent_mirrored(env, model, deterministic=False, N=N_test, perm=[-1, 0, 3, 2])
        print(f"Total test rew: {total_rew / N_test}")

        # Testing for permutation
        # N_test = 10
        # best_rew = 10000
        # best_perm = None
        # from itertools import permutations
        # for pos_perm in permutations(range(0, 4)):
        #     sign_perms = [[int(x) * 2 - 1 for x in list('{0:0b}'.format(i))] for i in range(16)]
        #     for sign_perm in sign_perms:
        #         pos_perm_copy = list(deepcopy(pos_perm))
        #         for i in range(len(sign_perm)):
        #             pos_perm_copy[-i] *= sign_perm[-i]
        #
        #         total_rew = test_agent_mirrored(env, model, deterministic=False, N=N_test, perm=pos_perm_copy)
        #         if total_rew < best_rew:
        #             best_rew = total_rew
        #             best_perm = pos_perm_copy
        # print(best_perm, best_rew)
