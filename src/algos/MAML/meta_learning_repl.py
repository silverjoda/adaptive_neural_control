import numpy as np
import torch as T
import torch.nn as nn
from copy import deepcopy

class SinTask:
    def __init__(self):
        self.a = np.random.rand() * 4 + 1
        self.b = np.random.rand() * np.pi


class SinPolicy(nn.Module):
    def __init__(self, hidden):
        super(SinPolicy, self).__init__()
        self.linear1 = nn.Linear(1, hidden)
        self.linear2 = nn.Linear(hidden, 1)


    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


def train_fomaml(env_fun, param_dict):
    # Initialize policy with meta parameters
    meta_policy = SinPolicy(param_dict["hidden"])
    meta_trn_opt = T.optim.SGD(meta_policy.parameters(), lr=0.01, momentum=0.9)

    for mt in range(param_dict["meta_training_iters"]):
        # Sample tasks
        env_list = [env_fun() for _ in range(param_dict["hidden"])]

        # Updated params list
        th_i_list = []

        for env in env_list:
            # Get data
            Xtrn, Ytrn, Xtst, Ytst = env.get_dataset()

            # Copy parameters to new network
            copied_meta_policy = deepcopy(meta_policy)

            # Evaluate gradient and updated parameter th_i on sampled task
            trn_opt = T.optim.SGD(copied_meta_policy.parameters(), lr=0.01, momentum=0.9)

            lossfun = nn.MSELoss()

            for t in range(param_dict["training_iters"]):
                Yhat = copied_meta_policy(Xtrn)
                loss = lossfun(Yhat, Ytrn)
                loss.backward()
                trn_opt.step()

            th_i_list.append(copied_meta_policy.parameters())


        # Calculate loss on test task


        # Update meta parameters

if __name__ == "__main__":
    param_dict = {"meta_training_iters" : 1000,
                  "training_iters": 1,
                  "hidden" : 24,
                  "batch_tasks" : 24,
                  "batch_trn" : 24}
    env_fun = SinTask
    train_fomaml(env_fun, param_dict)

