import numpy as np
import torch as T
import torch.nn as nn
from copy import deepcopy
import logging
import matplotlib.pyplot as plt

class SinTask:
    def __init__(self):
        self.a = np.random.rand() * 4 + 1
        self.b = np.random.rand() * np.pi

    def get_trn_data(self, n):
        self.X = np.linspace(0, 2 * np.pi, 2 * n).astype(np.float32)
        self.Y = self.a * np.sin(self.b + self.X)
        indeces = np.arange(n * 2)
        np.random.shuffle(indeces)
        self.trn_indeces = indeces[:n]
        self.tst_indeces = indeces[n:]
        return self.X[self.trn_indeces], self.Y[self.trn_indeces]

    def get_tst_data(self):
        return self.X[self.tst_indeces], self.Y[self.tst_indeces]

    def get_trn_tst_data_HC(self, n):
        Xtrn = np.linspace(0, 2 * np.pi, n).astype(np.float32)
        Xtst = np.linspace(np.pi, 2 * np.pi, n).astype(np.float32)
        Ytrn = self.a * np.sin(self.b + Xtrn)
        Ytst = self.a * np.sin(self.b + Xtst)
        return Xtrn, Xtst, Ytrn, Ytst

    def plot(self, *args, **kwargs):
        return plt.plot(self.X, self.Y, *args, **kwargs)


class SinPolicy(nn.Module):
    def __init__(self, hidden=24):
        super(SinPolicy, self).__init__()
        self.linear1 = nn.Linear(1, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, 1)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        h_relu = self.linear2(h_relu).clamp(min=0)
        y_pred = self.linear3(h_relu)
        return y_pred


def train_fomaml(env_fun, param_dict):
    # Initialize policy with meta parameters
    meta_policy = SinPolicy(param_dict["hidden"])
    meta_trn_opt = T.optim.SGD(meta_policy.parameters(), lr=param_dict["lr_meta"], momentum=param_dict["momentum_meta"])
    lossfun = nn.MSELoss()

    for mt in range(param_dict["meta_training_iters"]):
        # Clear meta gradients
        meta_trn_opt.zero_grad()

        # Sample tasks
        env_list = [env_fun() for _ in range(param_dict["batch_tasks"])]

        # Updated params list
        copied_meta_policy_list = []

        trn_losses = []
        for env in env_list:
            # Get data
            Xtrn, Ytrn = env.get_trn_data(param_dict["batch_trn"])

            # Copy parameters to new network
            copied_meta_policy = deepcopy(meta_policy)

            # Evaluate gradient and updated parameter th_i on sampled task
            trn_opt = T.optim.SGD(copied_meta_policy.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"])

            for t in range(param_dict["training_iters"]):
                Yhat = copied_meta_policy(T.from_numpy(Xtrn).unsqueeze(1))
                loss = lossfun(Yhat, T.from_numpy(Ytrn).unsqueeze(1))
                trn_losses.append(loss.detach().numpy())
                loss.backward()
                trn_opt.step()

            copied_meta_policy_list.append(copied_meta_policy)

        tst_losses = []
        # Calculate loss on test task
        for env, policy_i in zip(env_list, copied_meta_policy_list):
            Xtst, Ytst = env.get_tst_data()
            Yhat = policy_i(T.from_numpy(Xtst).unsqueeze(1))
            loss = lossfun(Yhat, T.from_numpy(Ytst).unsqueeze(1))
            tst_losses.append(loss.detach().numpy())
            loss.backward()

            # Add to meta gradients
            with T.no_grad():
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

    # Test the meta learned policy to adapt to a new task after n gradient steps
    env = env_fun()

    if False:
        Xtrn, Ytrn = env.get_trn_data(param_dict["batch_trn"])
        # Do training and evaluation on normal dataset
        policy_normal = deepcopy(meta_policy)
        opt = T.optim.SGD(policy_normal.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"])
        for t in range(param_dict["training_iters"]):
            Yhat = policy_normal(T.from_numpy(Xtrn).unsqueeze(1))
            loss = lossfun(Yhat, T.from_numpy(Ytrn).unsqueeze(1))
            loss.backward()
            opt.step()

        Yhat_tst = policy_normal(T.from_numpy(env.X).unsqueeze(1)).detach().numpy()

    if True:
        Xtrn, Ytrn, Xtst, Ytst = env.get_trn_tst_data_HC(param_dict["batch_trn"])
        # Do training and evaluation on hardcore dataset
        policy_normal = deepcopy(meta_policy)
        opt = T.optim.SGD(policy.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"])
        for t in range(param_dict["training_iters"]):
            Yhat = policy_normal(T.from_numpy(Xtrn).unsqueeze(1))
            loss = lossfun(Yhat, T.from_numpy(Ytrn).unsqueeze(1))
            loss.backward()
            opt.step()

        env.get_trn_data(param_dict["batch_trn"])
        Yhat_tst = policy_normal(T.from_numpy(env.X).unsqueeze(1)).detach().numpy()


    env.plot()
    plt.plot(env.X, Yhat_tst, "r")
    plt.show()

if __name__ == "__main__":
    policy = SinPolicy(24)

    param_dict = {"meta_training_iters" : 1000,
                  "training_iters": 3,
                  "hidden" : 24,
                  "batch_tasks" : 24,
                  "batch_trn" : 16,
                  "lr" : 0.01,
                  "lr_meta" : 0.003,
                  "momentum_trn" : 0.95,
                  "momentum_meta" : 0.95}

    env_fun = SinTask
    train_fomaml(env_fun, param_dict)

