import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np
from copy import deepcopy


class RNN_V3_LN_PG(nn.Module):
    def __init__(self, env, hid_dim=64, memory_dim=64, n_temp=3, tanh=False, to_gpu=False):
        super(RNN_V3_LN_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.hid_dim = hid_dim
        self.memory_dim = memory_dim
        self.tanh = tanh
        self.to_gpu = to_gpu

        self.rnn = nn.LSTM(self.obs_dim, self.memory_dim, n_temp, batch_first=True)
        self.fc1 = nn.Linear(self.obs_dim, self.obs_dim)
        self.m1 = nn.LayerNorm(self.obs_dim)
        self.fc2 = nn.Linear(self.obs_dim + self.memory_dim, self.hid_dim)
        self.m2 = nn.LayerNorm(self.hid_dim)
        self.fc3 = nn.Linear(self.hid_dim, self.act_dim)

        if to_gpu:
            self.log_std_gpu = T.zeros(1, self.act_dim).cuda()
        else:
            self.log_std_cpu = T.zeros(1, self.act_dim)


    def print_info(self):
        pass


    def soft_clip_grads(self, bnd=0.5):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            if p.grad is None: continue
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            #print("Soft clipping grads")

            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def forward(self, input):
        x, h = input
        rnn_features = F.selu(self.m1(self.fc1(x)))

        output, h = self.rnn(rnn_features, h)

        f = F.selu(self.m2(self.fc2(T.cat((output, x), 2))))
        if self.tanh:
            f = T.tanh(self.fc3(f))
        else:
            f = self.fc3(f)
        return f, h


    def sample_action(self, s):
        x, h = self.forward(s)
        return T.normal(x[0], T.exp(self.log_std_cpu)), h


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means, _ = self.forward((batch_states, None))

        # Calculate probabilities
        if self.to_gpu:
            log_std_batch = self.log_std_gpu.expand_as(action_means)
        else:
            log_std_batch = self.log_std_cpu.expand_as(action_means)

        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(2, keepdim=True)


class Linear_PG(nn.Module):
    def __init__(self, env, hid_dim=64, tanh=False, std_fixed=True, obs_dim=None, act_dim=None):
        super(Linear_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        if obs_dim is not None:
            self.obs_dim = obs_dim

        if act_dim is not None:
            self.act_dim = act_dim

        self.fc1 = nn.Linear(self.obs_dim, self.act_dim)

        if std_fixed:
            self.log_std = T.zeros(1, self.act_dim)
        else:
            self.log_std = nn.Parameter(T.zeros(1, self.act_dim))


    def forward(self, x):
        x = self.fc1(x)
        return x


    def soft_clip_grads(self, bnd=1):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m

        if maxval > bnd:
            # print("Soft clipping grads")
            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class NN_PG(nn.Module):
    def __init__(self, env, hid_dim=64, tanh=False, std_fixed=True, obs_dim=None, act_dim=None):
        super(NN_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        if obs_dim is not None:
            self.obs_dim = obs_dim

        if act_dim is not None:
            self.act_dim = act_dim

        self.tanh = tanh

        self.fc1 = nn.Linear(self.obs_dim, hid_dim)
        self.m1 = nn.LayerNorm(hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.m2 = nn.LayerNorm(hid_dim)
        self.fc3 = nn.Linear(hid_dim, self.act_dim)

        T.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='linear')

        if std_fixed:
            self.log_std = T.zeros(1, self.act_dim)
        else:
            self.log_std = nn.Parameter(T.zeros(1, self.act_dim))


    def forward(self, x):
        x = F.leaky_relu(self.m1(self.fc1(x)))
        x = F.leaky_relu(self.m2(self.fc2(x)))
        if self.tanh:
            x = T.tanh(self.fc3(x))
        else:
            x = self.fc3(x)
        return x


    def soft_clip_grads(self, bnd=1):
        # Find maximum
        maxval = 0

        for p in self.parameters():
            m = T.abs(p.grad).max()
            if m > maxval:
                maxval = m


        if maxval > bnd:
            # print("Soft clipping grads")
            for p in self.parameters():
                if p.grad is None: continue
                p.grad = (p.grad / maxval) * bnd

    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class NN_PG_VF(nn.Module):
    def __init__(self, env, hid_dim=64):
        super(NN_PG_VF, self).__init__()

        self.obs_dim = env.obs_dim + 1
        self.hid_dim = hid_dim

        self.fc1 = nn.Linear(self.obs_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, 1)

        T.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='linear')


    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

