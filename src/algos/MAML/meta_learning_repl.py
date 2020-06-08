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
        self.dataset_generated = False


    def get_dataset(self, n):
        if not self.dataset_generated:
            self.X = np.linspace(0, 2 * np.pi, 2 * n).astype(np.float32)
            self.Y = self.a * np.sin(self.b + self.X)
            indeces = np.arange(n * 2)
            np.random.shuffle(indeces)
            self.trn_indeces = indeces[:n]
            self.tst_indeces = indeces[n:]
            self.X_trn = self.X[self.trn_indeces]
            self.Y_trn = self.Y[self.trn_indeces]
            self.X_tst = self.X[self.tst_indeces]
            self.Y_tst = self.Y[self.tst_indeces]
            self.dataset_generated = True

        return self.X_trn, self.Y_trn, self.X_tst, self.Y_tst


    def get_dataset_halfsin(self, n):
        Xtrn = np.linspace(0, 2 * np.pi, n).astype(np.float32)
        Xtst = np.linspace(np.pi, 2 * np.pi, n).astype(np.float32)
        Ytrn = self.a * np.sin(self.b + Xtrn)
        Ytst = self.a * np.sin(self.b + Xtst)
        return Xtrn, Ytrn, Xtst, Ytst


    def plot_trn_set(self):
        if not self.dataset_generated:
            self.get_dataset(30)
        return plt.plot(self.X_trn, self.Y_trn, 'ro', self.X_tst, self.Y_tst, 'go')


    def plot_trn_set_halfsin(self):
        X_trn, Y_trn, X_tst, Y_tst = self.get_dataset_halfsin(15)
        return plt.plot(X_trn, Y_trn, 'ro', X_tst, Y_tst, 'go')


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

def train_maml(env_fun, param_dict):
    # Initialize policy with meta parameters
    meta_policy = SinPolicy(param_dict["hidden"])
    meta_trn_opt = T.optim.SGD(meta_policy.parameters(), lr=param_dict["lr_meta"], momentum=param_dict["momentum_meta"], weight_decay=param_dict["w_decay_meta"])
    lossfun = nn.MSELoss()

    for mt in range(param_dict["meta_training_iters"]):
        # Clear meta gradients
        meta_trn_opt.zero_grad()

        # Sample tasks
        env_list = [env_fun() for _ in range(param_dict["batch_tasks"])]
        dataset_list = [env.get_dataset(param_dict["batch_trn"]) for env in env_list]

        # Updated params list
        copied_meta_policy_list = []

        trn_losses = []
        for ei, env in enumerate(env_list):
            # Get data
            Xtrn, Ytrn, _, _ = dataset_list[ei]

            # Copy parameters to new network
            copied_meta_policy = deepcopy(meta_policy)
            copied_meta_policy_list.append(copied_meta_policy)

            # Evaluate gradient and updated parameter th_i on sampled task
            trn_opt = T.optim.SGD(copied_meta_policy.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])

            for t in range(param_dict["training_iters"]):
                Yhat = copied_meta_policy(T.from_numpy(Xtrn).unsqueeze(1))
                loss = lossfun(Yhat, T.from_numpy(Ytrn).unsqueeze(1))
                trn_losses.append(loss.detach().numpy())
                trn_opt.zero_grad()
                loss.backward()
                trn_opt.step()

            trn_opt.zero_grad()

        tst_losses = []
        # Calculate loss on test task
        for env, policy_i, dataset in zip(env_list, copied_meta_policy_list, dataset_list):
            _, _, Xtst, Ytst = dataset
            Yhat = policy_i(T.from_numpy(Xtst).unsqueeze(1))
            loss = lossfun(Yhat, T.from_numpy(Ytst).unsqueeze(1))
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
    opt_test_1 = T.optim.SGD(policy_test_1.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])
    policy_baseline_1 = SinPolicy(param_dict["hidden"])
    opt_baseline_1 = T.optim.SGD(policy_baseline_1.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])
    for t in range(param_dict["training_iters"]):
        Yhat = policy_test_1(T.from_numpy(Xtrn_1).unsqueeze(1))
        loss = lossfun(Yhat, T.from_numpy(Ytrn_1).unsqueeze(1))
        opt_test_1.zero_grad()
        loss.backward()
        opt_test_1.step()

        Yhat_baseline_1 = policy_baseline_1(T.from_numpy(Xtrn_1).unsqueeze(1))
        loss_baseline_1 = lossfun(Yhat_baseline_1, T.from_numpy(Ytrn_1).unsqueeze(1))
        opt_baseline_1.zero_grad()
        loss_baseline_1.backward()
        opt_baseline_1.step()


    Yhat_tst_1 = policy_test_1(T.from_numpy(Xtst_1).unsqueeze(1)).detach().numpy()
    Yhat_baseline_1 = policy_baseline_1(T.from_numpy(Xtst_1).unsqueeze(1)).detach().numpy()

    test_env_2 = env_fun()
    Xtrn_2, Ytrn_2, Xtst_2, Ytst_2 = test_env_2.get_dataset_halfsin(param_dict["batch_trn"])
    # Do training and evaluation on hardcore dataset
    policy_test_2 = deepcopy(meta_policy)
    opt_test_2 = T.optim.SGD(policy_test_2.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])
    policy_baseline_2 = SinPolicy(param_dict["hidden"])
    opt_baseline_2 = T.optim.SGD(policy_baseline_2.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])
    for t in range(param_dict["training_iters"]):
        Yhat = policy_test_2(T.from_numpy(Xtrn_2).unsqueeze(1))
        loss = lossfun(Yhat, T.from_numpy(Ytrn_2).unsqueeze(1))
        opt_test_2.zero_grad()
        loss.backward()
        opt_test_2.step()

        Yhat_baseline_2 = policy_baseline_1(T.from_numpy(Xtrn_2).unsqueeze(1))
        loss_baseline_2 = lossfun(Yhat_baseline_2, T.from_numpy(Ytrn_2).unsqueeze(1))
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


def train_fomaml(env_fun, param_dict):
    # Initialize policy with meta parameters
    meta_policy = SinPolicy(param_dict["hidden"])
    meta_trn_opt = T.optim.SGD(meta_policy.parameters(), lr=param_dict["lr_meta"], momentum=param_dict["momentum_meta"], weight_decay=param_dict["w_decay_meta"])
    lossfun = nn.MSELoss()

    for mt in range(param_dict["meta_training_iters"]):
        # Clear meta gradients
        meta_trn_opt.zero_grad()

        # Sample tasks
        env_list = [env_fun() for _ in range(param_dict["batch_tasks"])]
        dataset_list = [env.get_dataset(param_dict["batch_trn"]) for env in env_list]

        # Updated params list
        copied_meta_policy_list = []

        trn_losses = []
        for ei, env in enumerate(env_list):
            # Get data
            Xtrn, Ytrn, _, _ = dataset_list[ei]

            # Copy parameters to new network
            copied_meta_policy = deepcopy(meta_policy)
            copied_meta_policy_list.append(copied_meta_policy)

            # Evaluate gradient and updated parameter th_i on sampled task
            trn_opt = T.optim.SGD(copied_meta_policy.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])

            for t in range(param_dict["training_iters"]):
                Yhat = copied_meta_policy(T.from_numpy(Xtrn).unsqueeze(1))
                loss = lossfun(Yhat, T.from_numpy(Ytrn).unsqueeze(1))
                trn_losses.append(loss.detach().numpy())
                trn_opt.zero_grad()
                loss.backward()
                trn_opt.step()

            trn_opt.zero_grad()

        tst_losses = []
        # Calculate loss on test task
        for env, policy_i, dataset in zip(env_list, copied_meta_policy_list, dataset_list):
            _, _, Xtst, Ytst = dataset
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

    plt.ion()

    # Test the meta learned policy to adapt to a new task after n gradient steps
    test_env_1 = env_fun()
    Xtrn_1, Ytrn_1, Xtst_1, Ytst_1 = test_env_1.get_dataset(param_dict["batch_trn"])
    # Do training and evaluation on normal dataset
    policy_test_1 = deepcopy(meta_policy)
    opt_test_1 = T.optim.SGD(policy_test_1.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])
    policy_baseline_1 = SinPolicy(param_dict["hidden"])
    opt_baseline_1 = T.optim.SGD(policy_baseline_1.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])
    for t in range(param_dict["training_iters"]):
        Yhat = policy_test_1(T.from_numpy(Xtrn_1).unsqueeze(1))
        loss = lossfun(Yhat, T.from_numpy(Ytrn_1).unsqueeze(1))
        opt_test_1.zero_grad()
        loss.backward()
        opt_test_1.step()

        Yhat_baseline_1 = policy_baseline_1(T.from_numpy(Xtrn_1).unsqueeze(1))
        loss_baseline_1 = lossfun(Yhat_baseline_1, T.from_numpy(Ytrn_1).unsqueeze(1))
        opt_baseline_1.zero_grad()
        loss_baseline_1.backward()
        opt_baseline_1.step()


    Yhat_tst_1 = policy_test_1(T.from_numpy(Xtst_1).unsqueeze(1)).detach().numpy()
    Yhat_baseline_1 = policy_baseline_1(T.from_numpy(Xtst_1).unsqueeze(1)).detach().numpy()

    test_env_2 = env_fun()
    Xtrn_2, Ytrn_2, Xtst_2, Ytst_2 = test_env_2.get_dataset_halfsin(param_dict["batch_trn"])
    # Do training and evaluation on hardcore dataset
    policy_test_2 = deepcopy(meta_policy)
    opt_test_2 = T.optim.SGD(policy_test_2.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])
    policy_baseline_2 = SinPolicy(param_dict["hidden"])
    opt_baseline_2 = T.optim.SGD(policy_baseline_2.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])
    for t in range(param_dict["training_iters"]):
        Yhat = policy_test_2(T.from_numpy(Xtrn_2).unsqueeze(1))
        loss = lossfun(Yhat, T.from_numpy(Ytrn_2).unsqueeze(1))
        opt_test_2.zero_grad()
        loss.backward()
        opt_test_2.step()

        Yhat_baseline_2 = policy_baseline_1(T.from_numpy(Xtrn_2).unsqueeze(1))
        loss_baseline_2 = lossfun(Yhat_baseline_2, T.from_numpy(Ytrn_2).unsqueeze(1))
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


def train_reptile(env_fun, param_dict):
    # Initialize policy with meta parameters
    reptile_policy = SinPolicy(param_dict["hidden"])
    opt = T.optim.SGD(reptile_policy.parameters(), lr=param_dict["lr_meta"], momentum=param_dict["momentum_meta"], weight_decay=param_dict["w_decay_meta"])
    lossfun = nn.MSELoss()

    for rt in range(param_dict["reptile_training_iters"]):

        # Sample tasks
        env_list = [env_fun() for _ in range(param_dict["batch_tasks"])]
        dataset_list = [env.get_dataset(param_dict["batch_trn"]) for env in env_list]

        # Clear gradients
        opt.zero_grad()

        # Updated params list
        copied_policy_list = []

        trn_losses = []
        for ei, env in enumerate(env_list):
            # Get data
            Xtrn, Ytrn, _, _ = dataset_list[ei]

            # Copy parameters to new network
            copied_policy = deepcopy(reptile_policy)
            copied_policy_list.append(copied_policy)

            # Evaluate gradient and updated parameter th_i on sampled task
            trn_opt = T.optim.SGD(copied_policy.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])

            for t in range(param_dict["reptile_k"]):
                Yhat = copied_policy(T.from_numpy(Xtrn).unsqueeze(1))
                loss = lossfun(Yhat, T.from_numpy(Ytrn).unsqueeze(1))
                trn_losses.append(loss.detach().numpy())
                trn_opt.zero_grad()
                loss.backward()
                trn_opt.step()

        # Gather the weights and perform update
        reptile_policy_tmp = deepcopy(reptile_policy)
        for policy_i in copied_policy_list:
            # Add to meta gradients
            with T.no_grad():
                for p_r, p_r_t, p_i in zip(reptile_policy.parameters(), reptile_policy_tmp.parameters(), policy_i.parameters()):
                    p_r_t.data += param_dict["lr_reptile"] * ((p_i.data.clone() - p_r.data.clone()) / param_dict["batch_tasks"])

        # Copy weights
        for p_r, p_r_t in zip(reptile_policy.parameters(), reptile_policy_tmp.parameters()):
            p_r.data = p_r_t.data.clone()
        print("Iter: {}/{}, trn_mean_loss: {}".format(rt, param_dict["reptile_training_iters"], np.mean(trn_losses)))

    plt.ion()

    # Test the meta learned policy to adapt to a new task after n gradient steps
    test_env_1 = env_fun()
    Xtrn_1, Ytrn_1, Xtst_1, Ytst_1 = test_env_1.get_dataset(param_dict["batch_trn"])
    # Do training and evaluation on normal dataset
    policy_test_1 = deepcopy(reptile_policy)
    opt_test_1 = T.optim.SGD(policy_test_1.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])
    policy_baseline_1 = SinPolicy(param_dict["hidden"])
    opt_baseline_1 = T.optim.SGD(policy_baseline_1.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])
    for t in range(param_dict["reptile_k"]):
        Yhat = policy_test_1(T.from_numpy(Xtrn_1).unsqueeze(1))
        loss = lossfun(Yhat, T.from_numpy(Ytrn_1).unsqueeze(1))
        opt_test_1.zero_grad()
        loss.backward()
        opt_test_1.step()

        Yhat_baseline_1 = policy_baseline_1(T.from_numpy(Xtrn_1).unsqueeze(1))
        loss_baseline_1 = lossfun(Yhat_baseline_1, T.from_numpy(Ytrn_1).unsqueeze(1))
        opt_baseline_1.zero_grad()
        loss_baseline_1.backward()
        opt_baseline_1.step()


    Yhat_tst_1 = policy_test_1(T.from_numpy(Xtst_1).unsqueeze(1)).detach().numpy()
    Yhat_baseline_1 = policy_baseline_1(T.from_numpy(Xtst_1).unsqueeze(1)).detach().numpy()

    test_env_2 = env_fun()
    Xtrn_2, Ytrn_2, Xtst_2, Ytst_2 = test_env_2.get_dataset_halfsin(param_dict["batch_trn"])
    # Do training and evaluation on hardcore dataset
    policy_test_2 = deepcopy(reptile_policy)
    opt_test_2 = T.optim.SGD(policy_test_2.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])
    policy_baseline_2 = SinPolicy(param_dict["hidden"])
    opt_baseline_2 = T.optim.SGD(policy_baseline_2.parameters(), lr=param_dict["lr"], momentum=param_dict["momentum_trn"], weight_decay=param_dict["w_decay"])
    for t in range(param_dict["reptile_k"]):
        Yhat = policy_test_2(T.from_numpy(Xtrn_2).unsqueeze(1))
        loss = lossfun(Yhat, T.from_numpy(Ytrn_2).unsqueeze(1))
        opt_test_2.zero_grad()
        loss.backward()
        opt_test_2.step()

        Yhat_baseline_2 = policy_baseline_1(T.from_numpy(Xtrn_2).unsqueeze(1))
        loss_baseline_2 = lossfun(Yhat_baseline_2, T.from_numpy(Ytrn_2).unsqueeze(1))
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


if __name__ == "__main__":
    policy = SinPolicy(24)

    param_dict = {"meta_training_iters" : 800,
                  "reptile_training_iters": 1500,
                  "reptile_k" : 3,
                  "training_iters": 1, # 3
                  "hidden" : 24, # 24
                  "batch_tasks" : 24, # 24
                  "batch_trn" : 16, # 16
                  "lr" : 0.01, # 0.01
                  "lr_meta" : 0.001, # 0.001
                  "lr_reptile" : 0.03,
                  "momentum_trn" : 0.95, # 0.95
                  "momentum_meta" : 0.95, # 0.95
                  "w_decay" : 0.001, # 0.001
                  "w_decay_meta" : 0.001} # 0.001

    env_fun = SinTask
    #train_fomaml(env_fun, param_dict)
    train_reptile(env_fun, param_dict)
