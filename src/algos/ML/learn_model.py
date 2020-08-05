import random
import string
import time

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines import A2C
import pybullet as p

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


def learn_model(params, env, policy, regressor, gather_dataset=True):
    if gather_dataset:
        print("Starting dataset gathering")
        episode_observations = []
        episode_actions = []
        for i in range(params["dataset_episodes"]):
            observations = []
            actions = []
            obs = env.reset()
            for j in range(params["max_steps"]):
                action, _states = policy.predict(obs, deterministic=True)
                action = action + np.random.randn(env.act_dim)
                observations.append(obs)
                actions.append(action)
                obs, reward, done, info = env.step(action)

                if done:
                    episode_observations.append(observations)
                    episode_actions.append(actions)
                    break

            if i % 100 == 0:
                print("Dataset gathering episode: {}/{}".format(i, params["dataset_episodes"]))

        print("Gathered dataset, saving")
        episode_observations = np.array(episode_observations)
        episode_actions = np.array(episode_actions)
        np.save("data/observations", episode_observations)
        np.save("data/actions", episode_actions)
    else:
        print("Loaded dataset")
        episode_observations = np.load("data/observations")
        episode_actions = np.load("data/actions")

    # Optimizer
    optim = T.optim.Adam(params=regressor.parameters(), lr=params["regressor_lr"])

    # Start training
    print("Starting training")
    N = episode_observations.shape[0]
    for i in range(params["training_iters"]):
        # Sample dataset
        rnd_vec = np.random.choice(range(N), params["batchsize"], replace=False)
        observations_batch = episode_observations[rnd_vec]
        actions_batch = episode_actions[rnd_vec]

        s_batch = observations_batch[:, :-1, :]
        s_next_batch = observations_batch[:, 1:, :]
        act_batch = actions_batch[:, :-1, :]

        X = T.tensor(np.concatenate((s_batch, act_batch), axis=2), dtype=T.float32)
        Y = T.tensor(s_next_batch, dtype=T.float32)

        # Forward pass
        Y_ = regressor(X)

        # Update
        optim.zero_grad()
        loss = F.mse_loss(Y_, Y)
        loss.backward()
        optim.step()

        if i % 100 == 0:
            print("Iter: {}/{}  loss: {}".format(i, params["training_iters"], loss))

    # Save trained model
    print("Saving trained model")
    T.save(regressor.state_dict(), "agents/{}".format(params["ID"]))

def evaluate_model(params, env, policy, regressor):
    for i in range(params["eval_episodes"]):
        obs = env.reset()
        h = None
        for j in range(params["max_steps"]):

            # Get action from policy
            action, _states = policy.predict((obs, h), deterministic=True)
            action = action + np.random.randn(env.act_dim)

            with T.no_grad():
                # Predict next step
                if "lstm" in policy.__class__.lower():
                    obs_pred, h = regressor.forward_step(T.cat((obs, action)).unsqueeze(0))
                else:
                    obs_pred, h = regressor.forward(T.tensor(obs).unsqueeze(0))

            # Step env and get next obs GT
            obs, reward, done, info = env.step(action)

            # Calculate error and plot info
            obs_err = np.mean(obs - obs_pred.numpy())

            p.removeAllUserDebugItems()
            p.addUserDebugText("Mean err: % 3.3f" % (obs_err), [-1, 0, 2], textColorRGB=[np.clip(obs_err, 0, 1),1,0])

            if done: break

        if i % 100 == 0:
            print("Dataset gathering episode: {}/{}".format(i, params["dataset_episodes"]))

if __name__ == "__main__":
    from src.envs.bullet_cartpole.hangpole_goal_cont_variable.hangpole_goal_cont_variable import HangPoleGoalContVariableBulletEnv as env_fun

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"dataset_episodes": 1000,
              "eval_episodes": 100,
              "training_iters" : 10000,
              "batchsize": 30,
              "max_steps": 200,
              "gamma": 0.99,
              "regressor_lr": 0.001,
              "weight_decay": 0.0001,
              "normalize_rewards": False,
              "animate": False,
              "variable_velocity": False,
              "train": True,
              "note": "",
              "ID": ID}

    print(params)
    TRAINED_POLICY = False

    env = env_fun(animate=params["animate"],
                  max_steps=params["max_steps"],
                  action_input=False,
                  latent_input=False)

    # Here random or loaded learned policy
    policy = A2C('MlpPolicy', env)
    if TRAINED_POLICY:
        policy_dir = "agents/xxx.zip"
        policy = A2C.load(policy_dir)  # 2Q5

    # TODO: Make evaluation of the trained regressor. Evaluation has to be accept any policy and regressor (add hidden state to rnn, btw).
    # TODO: Evaluation also has to visibly show accuracy at every step.
    # TODO: Make multiple policy and/or dropout training to gauge confidence.

    # TODO: Questions to be answered
    # TODO: Q1) Can you learn good model from random policy (how does it generalize to state distribution induced by trained policy)
    # TODO: Q2) Does confidence using droupout or ensembles work
    # TODO: Q3) Does RNN learn model for adapting param
    # TODO: Q4) Using model for learning policy (later)

    # Make regressor NN agent
    #regressor = PyTorchMlp(env.obs_dim + env.act_dim, 24, env.obs_dim)
    regressor = PyTorchLSTM(env.obs_dim + env.act_dim, 24, env.obs_dim)

    # Train the agent
    t1 = time.time()
    learn_model(params, env, policy, regressor, gather_dataset=True)
    t2 = time.time()
    print("Training time: {}".format(t2-t1))
    print(params)
    env.close()

    # Evaluate the agent
    env = env_fun(animate=True,
                  max_steps=params["max_steps"],
                  action_input=False,
                  latent_input=False)
    evaluate_model(params, env, policy, regressor)
