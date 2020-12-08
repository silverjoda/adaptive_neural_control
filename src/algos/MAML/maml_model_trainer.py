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

class PyTorchMlp(nn.Module):
    def __init__(self, n_inputs=30, n_hidden=24, n_actions=18):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_actions)
        self.activ_fn = nn.Tanh()
        self.out_activ = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.activ_fn(self.fc1(x))
        x = self.activ_fn(self.fc2(x))
        x = self.fc3(x)
        return x

class PyTorchLSTM(nn.Module):
    def __init__(self, n_inputs=30, n_hidden=24, n_actions=18):
        nn.Module.__init__(self)

        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.LSTM(n_hidden, n_hidden, batch_first=True)
        self.fc3 = nn.Linear(n_hidden, n_actions)
        self.activ_fn = nn.Tanh()
        self.out_activ = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.activ_fn(self.fc1(x))
        x, h = self.fc2(x)
        x = self.fc3(x)
        return x

    def forward_step(self, x, h):
        x = self.activ_fn(self.fc1(x))
        x, h = self.fc2(x, h)
        x = self.fc3(x)
        return x, h

class MAMLModelTrainer:
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
    policy = PyTorchMlp(24)
    from src.envs.bullet_cartpole_archive.hangpole_goal_cont_variable.hangpole_goal_cont_variable import \
        HangPoleGoalContVariableBulletEnv as env_fun

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"meta_training_iters" : 25000,
                  "reptile_training_iters": 1500,
                  "reptile_k" : 5,
                  "training_iters": 1, # 3
                  "hidden" : 24, # 24
                  "batch_tasks" : 1, # 24
                  "batch_trn" : 16, # 16
                  "lr" : 0.01, # 0.01
                  "lr_meta" : 0.002, # 0.001
                  "lr_reptile" : 0.04,
                  "momentum_trn" : 0.95, # 0.95
                  "momentum_meta" : 0.95, # 0.95
                  "w_decay" : 0.001, # 0.001
                  "w_decay_meta" : 0.001,
                  "ID" : ID}

    env = env_fun(animate=False,
                  max_steps=200,
                  action_input=False,
                  latent_input=False,
                  is_variable=True)

    rollout_policy = A2C('MlpPolicy', env)
    meta_policy = PyTorchMlp(n_inputs=env.obs_dim, n_hidden=25, n_actions=env.act_dim)
    maml_model_trainer = MAMLModelTrainer(env, rollout_policy, params)
    meta_trained_policy = maml_model_trainer.meta_train_model(params, meta_policy)
    T.save(meta_trained_policy.state_dict(), "meta_agents/meta_regressor_{}".format(params["ID"]))
