import cma
import logging
import os
import random
import socket
import string
import sys
import time
import argparse

import cma
import torch
import torch as T
import yaml
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import optuna
import numpy as np

import src.my_utils as my_utils
import src.policies as policies

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
from torch.utils.tensorboard import SummaryWriter
T.set_num_threads(1)

def f_wrapper(env, policy):
    def f(w):
        reward = 0
        done = False
        obs = env.reset()

        vector_to_parameters(torch.from_numpy(w).float(), policy.parameters())

        while not done:
            with torch.no_grad():
                act = policy.sample_action(obs)
            obs, rew, done, _ = env.step(act)
            reward += rew

        return -reward
    return f

def f_optuna(trial, env, policy):
    reward = 0
    done = False
    obs = env.reset()

    w = parameters_to_vector(policy.parameters()).detach().numpy()

    w_suggested_increment = [trial.suggest_uniform(f'w_{i}', -0.05, 0.05) for i in range(6)]
    w = np.array(w_suggested_increment) + w

    vector_to_parameters(torch.from_numpy(w).float(), policy.parameters())

    while not done:
        with torch.no_grad():
            act = policy.sample_action(obs)
        obs, rew, done, _ = env.step(act)
        reward += rew

    return reward

def train(env, policy, config):
    w = parameters_to_vector(policy.parameters()).detach().numpy()
    es = cma.CMAEvolutionStrategy(w, config["cma_std"])
    f = f_wrapper(env, policy)

    sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        f'agents/{config["session_ID"]}_ES_policy.p')

    print(f'N_params: {len(w)}')

    it = 0
    try:
        while not es.stop():
            it += 1
            if it > config["iters"]:
                break
            X = es.ask()
            es.tell(X, [f(x) for x in X])
            es.disp()

    except KeyboardInterrupt:
        print("User interrupted process.")

    vector_to_parameters(torch.from_numpy(es.result.xbest).float(), policy.parameters())
    T.save(policy.state_dict(), sdir)
    print("Saved agent, {}".format(sdir))

    return es.result.fbest


def train_optuna(env, policy, config):
    w = parameters_to_vector(policy.parameters()).detach().numpy()

    study = optuna.create_study(direction='maximize')
    sdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        f'agents/{config["session_ID"]}_ES_policy.p')

    print(f'N_params: {len(w)}')

    study.optimize(lambda trial : f_optuna(trial, env, policy), n_trials=config["optuna_trials"], show_progress_bar=True)

    vector_to_parameters(torch.from_numpy(np.array([0,0,0,0] + list(study.best_params.values()))).float(), policy.parameters())
    T.save(policy.state_dict(), sdir)
    print("Saved agent, {}, with best value: {}".format(sdir, study.best_value))

    return study.best_value


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
    parser.add_argument('--algo_config', type=str, default="configs/es_default_config.yaml", required=False,
                        help='Algorithm config file name. ')
    parser.add_argument('--env_config', type=str, default="default.yaml", required=False,
                        help='Env config file name. ')
    parser.add_argument('--iters', type=int, required=False, default=200000, help='Number of training steps. ')

    args = parser.parse_args()
    return args.__dict__

def read_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def test_agent(env, policy, N=30):
    total_rew = 0
    for _ in range(N):
        obs = env.reset()
        cum_rew = 0
        while True:
            action = policy.sample_action(obs)
            obs, reward, done, info = env.step(action)
            cum_rew += reward
            total_rew += reward
            if done:
                print(cum_rew)
                break
    return total_rew / N

def test_agent_draw(env, policy, N=30):
    import matplotlib.pyplot as plt
    total_rew = 0
    for _ in range(N):
        obs = env.reset()
        cum_rew = 0
        leg_pts = []
        step_ctr = 0
        while True:
            action = policy.sample_action(obs)
            obs, reward, done, info = env.step(action)
            cum_rew += reward
            total_rew += reward

            _, _, torso_vel, _, joint_angles, _, _, _, _ = env.get_obs()

            if step_ctr > 10:
                leg_pts.append(env.single_leg_dkt(joint_angles[9:12]))

            if step_ctr == 60:
                break
            step_ctr += 1

            if done:
                print(cum_rew)
                break

        x = [leg_pt[0] for leg_pt in leg_pts]
        z = [leg_pt[2] for leg_pt in leg_pts]
        colors = np.random.rand(len(leg_pts))
        plt.scatter(x, z, c=colors, alpha=0.5)
        plt.show()

def test_agent_adapt(env, policy, N=30):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial : f_optuna(trial, env, policy), n_trials=20, show_progress_bar=True)

    total_rew = 0
    for _ in range(N):
        obs = env.reset()
        cum_rew = 0
        while True:
            action = policy.sample_action(obs)
            obs, reward, done, info = env.step(action)
            cum_rew += reward
            total_rew += reward
            if done:
                print(cum_rew)
                break

    return total_rew / N

if __name__=="__main__":
    args = parse_args()
    algo_config = read_config(args["algo_config"])
    env_config = read_config(args["env_config"])
    config = {**args, **algo_config, **env_config}

    print(config)

    for s in ["agents", "agents_cp", "tb"]:
        if not os.path.exists(s):
            os.makedirs(s)

        # Random ID of this session
        if config["default_session_ID"] is None:
            config["session_ID"] = ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3))
        else:
            config["session_ID"] = "TST"

    # Import correct env by name
    env_fun = my_utils.import_env(config["env_name"])
    env = env_fun(config)

    policy = my_utils.make_policy(env, config)

    if config["train"] or socket.gethostname() == "goedel":
        if config["preload"]:
            policy.load_state_dict(T.load(config["test_agent_path"]))

        t1 = time.time()
        if config["algo"] == "cma":
            train(env, policy, config)
        elif config["algo"] == "optuna":
            train_optuna(env, policy, config)
        else:
            print("Algorithm not implemented")
            exit()

        t2 = time.time()

        print("Training time: {}".format(t2 - t1))
        print(config)

    if config["test"] and socket.gethostname() != "goedel":
        if not args["train"]:
            policy.load_state_dict(T.load(config["test_agent_path"]))
        print([par.item() for par in policy.parameters()])
        avg_rew = test_agent(env, policy)
        print(f"Avg test rew: {avg_rew}")

        # On tiles policy:
        # Tiles: ~50
        # Flat: ~58
        # Perlin: ~52


