import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np
from copy import deepcopy

class NN_PG(nn.Module):
    def __init__(self, env, config):
        super(NN_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.tanh = config["policy_lastlayer_tanh"]
        self.hid_dim = config["policy_hid_dim"]
        self.grad_clip_value = config["policy_grad_clip_value"]
        self.policy_residual_connection = config["policy_residual_connection"]
        self.activation_fun = F.leaky_relu

        self.fc1 = nn.Linear(self.obs_dim, self.hid_dim)
        self.m1 = nn.LayerNorm(self.hid_dim)
        self.fc2 = nn.Linear(self.hid_dim, self.hid_dim)
        self.m2 = nn.LayerNorm(self.hid_dim)
        self.fc3 = nn.Linear(self.hid_dim, self.act_dim)

        if self.policy_residual_connection:
            self.fc_res = nn.Linear(self.obs_dim, self.act_dim)

        self.log_std = T.zeros(1, self.act_dim)

        T.nn.init.zeros_(self.fc1.bias)
        T.nn.init.zeros_(self.fc2.bias)
        T.nn.init.zeros_(self.fc3.bias)
        T.nn.init.kaiming_normal(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='linear')

        for p in self.parameters():
            p.register_hook(lambda grad: T.clamp(grad, -self.grad_clip_value, self.grad_clip_value))

    def forward(self, x):
        x = self.activation_fun(self.m1(self.fc1(x)))
        x = self.activation_fun(self.m2(self.fc2(x)))
        if self.policy_residual_connection:
            x = self.fc3(x) + self.fc_res(x)
        else:
            x = self.fc3(x)
        if self.tanh:
            x = T.tanh(x)
        return x

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

class RNN_PG(nn.Module):
    def __init__(self, env, config):
        super(RNN_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.tanh = config["policy_lastlayer_tanh"]
        self.hid_dim = config["policy_hid_dim"]
        self.policy_grad_clip_value = config["policy_grad_clip_value"]
        self.n_lstm_temporal_layers = config["policy_n_lstm_temporal_layers"]
        self.activation_fun = F.leaky_relu

        self.fc_x_h_1 = nn.Linear(self.obs_dim, self.memory_dim)
        self.fc_x_h_2 = nn.Linear(self.obs_dim, self.memory_dim)
        self.fc_x_h_3 = nn.Linear(self.obs_dim, self.memory_dim)

        self.rnn_1 = nn.LSTM(self.memory_dim, self.memory_dim, batch_first=True)
        self.m_rnn_1 = nn.LayerNorm(self.memory_dim)
        self.rnn_2 = nn.LSTM(self.memory_dim, self.memory_dim, batch_first=True)
        self.m_rnn_2 = nn.LayerNorm(self.memory_dim)
        self.rnn_3 = nn.LSTM(self.memory_dim, self.memory_dim, batch_first=True)

        self.fc_h_y_1 = nn.Linear(self.memory_dim, self.memory_dim)
        self.fc_h_y_2 = nn.Linear(self.memory_dim, self.memory_dim)
        self.fc_h_y_3 = nn.Linear(self.memory_dim, self.memory_dim)
        self.fc_res = nn.Linear(self.obs_dim, self.memory_dim)

        self.log_std = T.zeros(1, self.act_dim)

        T.nn.init.zeros_(self.fc1.bias)
        T.nn.init.zeros_(self.fc2.bias)
        T.nn.init.zeros_(self.fc3.bias)
        T.nn.init.kaiming_normal(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='linear')

        for p in self.parameters():
            p.register_hook(lambda grad: T.clamp(grad, -self.grad_clip_value, self.grad_clip_value))

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
