import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import logging
import matplotlib.pyplot as plt
from stable_baselines import A2C
import random
import string
import time
import random
import string
import socket
import numpy as np
import argparse
import yaml
import os

#for p in model.parameters():
#    p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

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
    if name == "quadruped":
        from src.envs.bullet_nexabot.quadruped.quadruped import QuadrupedBulletEnv as env_fun
    if name == "quadrotor":
        from src.envs.bullet_quadrotor.quadrotor import QuadrotorBulletEnv as env_fun
    if name == "buggy":
        from src.envs.bullet_buggy.buggy import BuggyBulletEnv as env_fun
    assert env_fun is not None, "Env name not found, exiting. "
    return env_fun

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

class MAMLRLTrainer:
    def __init__(self, env, policy, params):
        self.env = env
        self.policy = policy

    def get_env_dataset(self):
        episode_observations_trn = []
        episode_actions_trn = []
        episode_observations_tst = []
        episode_actions_tst = []
        for i in range(params["dataset_episodes"]):
            observations = []
            actions = []
            obs = env.reset(randomize=True)
            for j in range(params["max_steps"]):
                action, _states = policy.predict(obs, deterministic=True)
                action = action + np.random.randn(env.act_dim)
                observations.append(obs)
                actions.append(action)
                obs, reward, done, info = env.step(action)

                if done:
                    episode_observations_trn.append(observations)
                    episode_actions_trn.append(actions)
                    break

            observations = []
            actions = []
            obs = env.reset(randomize=False)
            for j in range(params["max_steps"]):
                action, _states = policy.predict(obs, deterministic=True)
                action = action + np.random.randn(env.act_dim)
                observations.append(obs)
                actions.append(action)
                obs, reward, done, info = env.step(action)

                if done:
                    episode_observations_tst.append(observations)
                    episode_actions_tst.append(actions)
                    break

        return episode_observations_trn, episode_actions_trn, episode_observations_tst, episode_actions_tst

    def meta_train_model(self, param_dict, meta_policy):
        meta_trn_opt = T.optim.SGD(meta_policy.parameters(),
                                   lr=param_dict["lr_meta"],
                                   momentum=param_dict["momentum_meta"],
                                   weight_decay=param_dict["w_decay_meta"])

        for mt in range(param_dict["meta_training_iters"]):
            # Clear meta gradients
            meta_trn_opt.zero_grad()

            # Sample tasks
            dataset_list = [self.get_env_dataset()]

            # Updated params list
            copied_meta_policy_list = []

            trn_losses = []
            for d in dataset_list:
                # Get data
                Xtrn, Ytrn, _, _ = d

                # Copy parameters to new network
                copied_meta_policy = deepcopy(meta_policy)
                copied_meta_policy_list.append(copied_meta_policy)

                # Evaluate gradient and updated parameter th_i on sampled task
                trn_opt = T.optim.SGD(copied_meta_policy.parameters(), lr=param_dict["lr"],
                                      momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])

                for t in range(param_dict["training_iters"]):
                    Yhat = copied_meta_policy(T.from_numpy(Xtrn).unsqueeze(1))
                    loss = F.mse_loss(Yhat, T.from_numpy(Ytrn).unsqueeze(1))
                    trn_losses.append(loss.detach().numpy())
                    trn_opt.zero_grad()
                    loss.backward(create_graph=True)
                    trn_opt.step()

            tst_losses = []
            # Calculate loss on test task
            for policy_i, dataset in zip(copied_meta_policy_list, dataset_list):
                _, _, Xtst, Ytst = dataset
                Yhat = policy_i(T.from_numpy(Xtst).unsqueeze(1))
                loss = F.mse_loss(Yhat, T.from_numpy(Ytst).unsqueeze(1))
                tst_losses.append(loss.detach().numpy())
                loss.backward()

                # Add to meta gradients
                for p1, p2 in zip(meta_policy.parameters(), policy_i.parameters()):
                    if p1.grad is None:
                        p1.grad = p2.grad.clone()
                    else:
                        p1.grad += p2.grad.clone()

            # Divide gradient by batchsize
            for p in meta_policy.parameters():
                p.grad /= param_dict["batch_tasks"]

            # Update meta parameters
            meta_trn_opt.step()

            print("Meta iter: {}/{}, trn_mean_loss: {}, tst_mean_loss: {}".format(mt,
                                                                                  param_dict["meta_training_iters"],
                                                                                  np.mean(trn_losses),
                                                                                  np.mean(tst_losses)))

        plt.ion()

        # Test the meta learned policy to adapt to a new task after n gradient steps
        test_env_1 = env_fun()
        Xtrn_1, Ytrn_1, Xtst_1, Ytst_1 = test_env_1.get_dataset(param_dict["batch_trn"])
        # Do training and evaluation on normal dataset
        policy_test_1 = deepcopy(meta_policy)
        opt_test_1 = T.optim.SGD(policy_test_1.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"],
                                 weight_decay=param_dict["w_decay"])
        policy_baseline_1 = SinPolicy(param_dict["hidden"])
        opt_baseline_1 = T.optim.SGD(policy_baseline_1.parameters(), lr=param_dict["lr"],
                                     momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])
        for t in range(param_dict["training_iters"]):
            Yhat = policy_test_1(T.from_numpy(Xtrn_1).unsqueeze(1))
            loss = F.mse_loss(Yhat, T.from_numpy(Ytrn_1).unsqueeze(1))
            opt_test_1.zero_grad()
            loss.backward()
            opt_test_1.step()

            Yhat_baseline_1 = policy_baseline_1(T.from_numpy(Xtrn_1).unsqueeze(1))
            loss_baseline_1 = F.mse_loss(Yhat_baseline_1, T.from_numpy(Ytrn_1).unsqueeze(1))
            opt_baseline_1.zero_grad()
            loss_baseline_1.backward()
            opt_baseline_1.step()

        Yhat_tst_1 = policy_test_1(T.from_numpy(Xtst_1).unsqueeze(1)).detach().numpy()
        Yhat_baseline_1 = policy_baseline_1(T.from_numpy(Xtst_1).unsqueeze(1)).detach().numpy()

        test_env_2 = env_fun()
        Xtrn_2, Ytrn_2, Xtst_2, Ytst_2 = test_env_2.get_dataset_halfsin(param_dict["batch_trn"])
        # Do training and evaluation on hardcore dataset
        policy_test_2 = deepcopy(meta_policy)
        opt_test_2 = T.optim.SGD(policy_test_2.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"],
                                 weight_decay=param_dict["w_decay"])
        policy_baseline_2 = SinPolicy(param_dict["hidden"])
        opt_baseline_2 = T.optim.SGD(policy_baseline_2.parameters(), lr=param_dict["lr"],
                                     momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])
        for t in range(param_dict["training_iters"]):
            Yhat = policy_test_2(T.from_numpy(Xtrn_2).unsqueeze(1))
            loss = F.mse_loss(Yhat, T.from_numpy(Ytrn_2).unsqueeze(1))
            opt_test_2.zero_grad()
            loss.backward()
            opt_test_2.step()

            Yhat_baseline_2 = policy_baseline_1(T.from_numpy(Xtrn_2).unsqueeze(1))
            loss_baseline_2 = F.mse_loss(Yhat_baseline_2, T.from_numpy(Ytrn_2).unsqueeze(1))
            opt_baseline_2.zero_grad()
            loss_baseline_2.backward()
            opt_baseline_2.step()

        Yhat_tst_2 = policy_test_2(T.from_numpy(Xtst_2).unsqueeze(1)).detach().numpy()
        Yhat_baseline_2 = policy_baseline_2(T.from_numpy(Xtst_2).unsqueeze(1)).detach().numpy()

        plt.figure()
        test_env_1.plot_trn_set()
        plt.plot(Xtst_1, Yhat_tst_1, "bo")
        plt.plot(Xtst_1, Yhat_baseline_1, "ko")
        plt.title("Env_1")
        plt.show()
        plt.pause(.001)

        plt.figure()
        plt.title("Env_2")
        test_env_2.plot_trn_set_halfsin()
        plt.plot(Xtst_2, Yhat_tst_2, "bo")
        plt.plot(Xtst_2, Yhat_baseline_2, "ko")
        plt.show()
        plt.pause(1000)

        return policy

if __name__ == "__main__":
    args = parse_args()
    algo_config = read_config(args["algo_config"])
    env_config = read_config(args["env_config"])
    aug_env_config = {**args, **env_config}

    print(args)
    print(algo_config)
    print(env_config)

    env_fun = import_env(env_config["env_name"])
    noise_fun = make_action_noise_fun(algo_config)

    session_ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))

    maml_rl_trainer = MAMLRLTrainer(algo_config, env_config, env_fun)
    maml_rl_trainer.meta_train(n_meta_iters=algo_config["n_meta_iters"])
    maml_rl_trainer.test()
