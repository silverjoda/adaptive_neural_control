import random
import string
import time
import itertools

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

class PyTorchMlpCst(nn.Module):

    def __init__(self, n_inputs=30, n_hidden=24, n_actions=18):
        nn.Module.__init__(self)
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_actions)
        self.activ_fn = nn.Tanh()
        self.out_activ = nn.Softmax(dim=0)

    def forward(self, x):
        return x[:, :, self.n_actions]

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

def learn_regressor(params, env, policy, regressor, gather_dataset=True):
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
            print("Iter: {}/{}  loss: {}".format(i + 1, params["training_iters"], loss))

    # Save trained model
    print("Saving trained model")
    T.save(regressor.state_dict(), "agents/regressor_{}".format(params["ID"]))

def evaluate_model(params, env, policy, regressor):
    mse_errors = []
    for i in range(params["eval_episodes"]):
        obs = env.reset()
        h = None
        mse_episode_errors = []
        for j in range(params["max_steps"]):

            # Get action from policy
            action, _states = policy.predict(obs, deterministic=True)
            action = action + np.random.randn(env.act_dim)

            with T.no_grad():
                # Predict next step
                if "lstm" in regressor.__class__.__name__.lower():
                    obs_pred, h = regressor.forward_step(T.tensor(np.concatenate((obs, action)), dtype=T.float32).unsqueeze(0).unsqueeze(0), h)
                    obs_pred = obs_pred[0]
                else:
                    obs_pred  = regressor.forward(T.tensor(np.concatenate((obs, action)), dtype=T.float32).unsqueeze(0))

            # Step env and get next obs GT
            obs, reward, done, info = env.step(action)

            # Calculate error and plot info
            obs_err = np.mean(np.square(obs - obs_pred.numpy()))
            mse_episode_errors.append(obs_err)

            p.removeAllUserDebugItems()
            clp_err = np.clip(obs_err, 0, 1)
            p.addUserDebugText("Mean err: % 3.3f" % (obs_err), [-1, 0, 2], textColorRGB=[clp_err,1-clp_err,1-clp_err], textSize=2)

            time.sleep(0.01)

            if done: print("Done, breaking"); break

        mse_errors.append(mse_episode_errors)

        print("Eval episode: {}/{}, mean_mse = {}, min_mse = {}, max_mse = {}".format(i,
                                                                                      params["eval_episodes"],
                                                                                      np.mean(mse_episode_errors),
                                                                                      np.min(mse_episode_errors),
                                                                                      np.max(mse_episode_errors)))

    mean_mse = np.mean(mse_errors)
    min_mse = np.min(mse_errors)
    max_mse = np.max(mse_errors)
    print("Evaluation complete, Global mean_mse = {}, Global min_mse = {}, Global max_mse = {}".format(mean_mse, min_mse, max_mse))
    return mean_mse, min_mse, max_mse

def run_experiment(params, LOAD_POLICY, LOAD_REGRESSOR, TRAIN_REGRESSOR, LSTM_POLICY, VARIABLE_TRAIN, VARIABLE_EVAL):
    env = env_fun(animate=params["animate"],
                  max_steps=params["max_steps"],
                  action_input=False,
                  latent_input=False,
                  is_variable=VARIABLE_TRAIN)

    # Here random or loaded learned policy
    policy = A2C('MlpPolicy', env)
    if LOAD_POLICY:
        policy_dir = "agents/xxx.zip"
        policy = A2C.load(policy_dir)  # 2Q5

    # Make regressor NN agent
    if LSTM_POLICY:
        regressor = PyTorchLSTM(env.obs_dim + env.act_dim, 24, env.obs_dim)
    else:
        regressor = PyTorchMlp(env.obs_dim + env.act_dim, 24, env.obs_dim)

    if TRAIN_REGRESSOR:
        # Train the regressor
        t1 = time.time()
        learn_regressor(params, env, policy, regressor, gather_dataset=True)
        t2 = time.time()
        print("Training time: {}".format(t2 - t1))
        print(params)
    env.close()

    if LOAD_REGRESSOR:
        regressor_dir = "agents/regressor_Y1G"
        regressor.load_state_dict(T.load(regressor_dir))

    # Evaluate the agent
    env = env_fun(animate=params["animate"],
                  max_steps=params["max_steps"],
                  action_input=False,
                  latent_input=False,
                  is_variable=VARIABLE_EVAL)
    return evaluate_model(params, env, policy, regressor)

def run_baseline(params, LOAD_POLICY, VARIABLE_EVAL):
    # Evaluate the agent
    env = env_fun(animate=params["animate"],
                  max_steps=params["max_steps"],
                  action_input=False,
                  latent_input=False,
                  is_variable=VARIABLE_EVAL)
    policy = A2C('MlpPolicy', env)
    if LOAD_POLICY:
        policy_dir = "agents/xxx.zip"
        policy = A2C.load(policy_dir)  # 2Q5
    regressor = PyTorchMlpCst(env.obs_dim + env.act_dim, 24, env.obs_dim)
    return evaluate_model(params, env, policy, regressor)

if __name__ == "__main__":
    from src.envs.bullet_cartpole_archive.hangpole_goal_cont_variable.hangpole_goal_cont_variable import HangPoleGoalContVariableBulletEnv as env_fun

    ID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    params = {"dataset_episodes": 30000,
              "training_iters": 30000,
              "eval_episodes": 100,
              "batchsize": 30,
              "gamma": 0.99,
              "regressor_lr": 0.001,
              "weight_decay": 0.0001,
              "max_steps": 200,
              "normalize_rewards": False,
              "animate": False,
              "note": "",
              "ID": ID}

    LOAD_POLICY = False
    LOAD_REGRESSOR = False
    TRAIN_REGRESSOR = True

    results = []

    for prod in itertools.product([0,1], [0,1], [0,1]):
        LSTM_POLICY, VARIABLE_TRAIN, VARIABLE_EVAL = prod
        mean_mse, min_mse, max_mse = run_experiment(params,
                                              LOAD_POLICY=LOAD_POLICY,
                                              LOAD_REGRESSOR=LOAD_REGRESSOR,
                                              TRAIN_REGRESSOR=TRAIN_REGRESSOR,
                                              LSTM_POLICY=LSTM_POLICY,
                                              VARIABLE_TRAIN=VARIABLE_TRAIN,
                                              VARIABLE_EVAL=VARIABLE_EVAL)
        results.append([LSTM_POLICY, VARIABLE_TRAIN, VARIABLE_EVAL, mean_mse, min_mse, max_mse])

    print("Final results ====================================================")
    print("==================================================================")

    # TODO: DEBUG AND TEST BASELINE EVALUATION
    mean_mse_cst, min_mse_cst, max_mse_cst = run_baseline(params, LOAD_POLICY=LOAD_POLICY, VARIABLE_EVAL=False)
    print("Constant Regressor (just predict previous state): VARIABLE_EVAL: False, Results: mean_mse: {}, min_mse: {}, max_mse: {}".format(
        mean_mse_cst, min_mse_cst, max_mse_cst))
    mean_mse_cst, min_mse_cst, max_mse_cst = run_baseline(params, LOAD_POLICY=LOAD_POLICY, VARIABLE_EVAL=True)
    print("Constant Regressor (just predict previous state): VARIABLE_EVAL: True, Results: mean_mse: {}, min_mse: {}, max_mse: {}".format(
        mean_mse_cst, min_mse_cst, max_mse_cst))

    for r in results:
        LSTM_POLICY, VARIABLE_TRAIN, VARIABLE_EVAL, mean_mse, min_mse, max_mse = r
        print(
            "Evaluating: LSTM_POLICY: {}, VARIABLE_TRAIN: {}, VARIABLE_EVAL: {}. Results: mean_mse: {}, min_mse: {}, max_mse: {}".
            format(LSTM_POLICY, VARIABLE_TRAIN, VARIABLE_EVAL, mean_mse, min_mse, max_mse))

    # TODO: Make evaluation of the trained regressor. Evaluation has to be accept any policy and regressor (add hidden state to rnn, btw).
    # TODO: Evaluation also has to visibly show accuracy at every step.
    # TODO: Make multiple policy and/or dropout training to gauge confidence.

    # TODO: Questions to be answered
    # TODO: Q1) Can you learn good model from random policy (how does it generalize to state distribution induced by trained policy)
    # TODO: Q2) Does confidence using droupout or ensembles work
    # TODO: Q3) Does RNN learn model for adapting param
    # TODO: Q4) Using model for learning policy (later)

