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
        T.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
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
        self.policy_memory_dim = config["policy_memory_dim"]
        self.policy_grad_clip_value = config["policy_grad_clip_value"]
        self.activation_fun = F.leaky_relu

        self.fc_in = nn.Linear(self.obs_dim, self.policy_memory_dim)
        self.rnn = nn.LSTM(self.policy_memory_dim, self.policy_memory_dim, 3, batch_first=True)
        self.fc_out = nn.Linear(self.policy_memory_dim, self.policy_memory_dim)
        self.fc_res = nn.Linear(self.obs_dim, self.policy_memory_dim)

        self.log_std = T.zeros(1, self.act_dim)
        self.activation_fun = F.leaky_relu

        T.nn.init.zeros_(self.fc_in.bias)
        T.nn.init.kaiming_normal_(self.fc_in.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.zeros_(self.fc_out.bias)
        T.nn.init.kaiming_normal_(self.fc_out.weight, mode='fan_in', nonlinearity='linear')

        T.nn.init.zeros_(self.rnn.bias_hh_l0)
        T.nn.init.zeros_(self.rnn.bias_ih_l0)
        T.nn.init.kaiming_normal_(self.rnn.weight_ih_l0, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.orthogonal_(self.rnn.weight_hh_l0)

        for p in self.parameters():
            p.register_hook(lambda grad: T.clamp(grad, -self.grad_clip_value, self.grad_clip_value))

    def forward(self, input):
        x, h = input
        rnn_features = self.activation_fun(self.fc1(x))

        rnn_output, h = self.rnn(rnn_features, h)

        y = self.fc_out(F.tanh(self.fc_res(x)) + rnn_output)
        if self.tanh:
            y = T.tanh(y)

        return y, h

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

class RNN_RES_PG(nn.Module):
    def __init__(self, env, config):
        super(RNN_RES_PG, self).__init__()
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.tanh = config["policy_lastlayer_tanh"]
        self.policy_memory_dim = config["policy_memory_dim"]
        self.policy_grad_clip_value = config["policy_grad_clip_value"]
        self.activation_fun = F.leaky_relu

        self.fc_x_h_1 = nn.Linear(self.obs_dim, self.policy_memory_dim)
        self.fc_x_h_2 = nn.Linear(self.obs_dim, self.policy_memory_dim)
        self.fc_x_h_3 = nn.Linear(self.obs_dim, self.policy_memory_dim)

        self.rnn_1 = nn.LSTM(self.policy_memory_dim, self.policy_memory_dim, batch_first=True)
        self.m_rnn_1 = nn.LayerNorm(self.policy_memory_dim)
        self.rnn_2 = nn.LSTM(self.policy_memory_dim, self.policy_memory_dim, batch_first=True)
        self.m_rnn_2 = nn.LayerNorm(self.policy_memory_dim)
        self.rnn_3 = nn.LSTM(self.policy_memory_dim, self.policy_memory_dim, batch_first=True)

        self.fc_h_y_1 = nn.Linear(self.policy_memory_dim, self.policy_memory_dim)
        self.fc_h_y_2 = nn.Linear(self.policy_memory_dim, self.policy_memory_dim)
        self.fc_h_y_3 = nn.Linear(self.policy_memory_dim, self.policy_memory_dim)
        self.fc_res = nn.Linear(self.obs_dim, self.policy_memory_dim)

        self.log_std = T.zeros(1, self.act_dim)
        self.activation_fun = F.leaky_relu

        T.nn.init.zeros_(self.fc_x_h_1.bias)
        T.nn.init.zeros_(self.fc_x_h_2.bias)
        T.nn.init.zeros_(self.fc_x_h_3.bias)
        T.nn.init.zeros_(self.fc_h_y_1.bias)
        T.nn.init.zeros_(self.fc_h_y_2.bias)
        T.nn.init.zeros_(self.fc_h_y_3.bias)
        T.nn.init.kaiming_normal_(self.fc_x_h_1.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc_x_h_2.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc_x_h_3.weight, mode='fan_in', nonlinearity='linear')
        T.nn.init.kaiming_normal_(self.fc_h_y_1.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc_h_y_2.weight, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.kaiming_normal_(self.fc_h_y_3.weight, mode='fan_in', nonlinearity='linear')

        T.nn.init.zeros_(self.rnn_1.bias_hh_l0)
        T.nn.init.zeros_(self.rnn_1.bias_ih_l0)
        T.nn.init.kaiming_normal_(self.rnn_1.weight_ih_l0, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.orthogonal_(self.rnn_1.weight_hh_l0)

        T.nn.init.zeros_(self.rnn_2.bias_hh_l0)
        T.nn.init.zeros_(self.rnn_2.bias_ih_l0)
        T.nn.init.kaiming_normal_(self.rnn_2.weight_ih_l0, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.orthogonal_(self.rnn_2.weight_hh_l0)

        T.nn.init.zeros_(self.rnn_3.bias_hh_l0)
        T.nn.init.zeros_(self.rnn_3.bias_ih_l0)
        T.nn.init.kaiming_normal_(self.rnn_3.weight_ih_l0, mode='fan_in', nonlinearity='leaky_relu')
        T.nn.init.orthogonal_(self.rnn_3.weight_hh_l0)

        for p in self.parameters():
            p.register_hook(lambda grad: T.clamp(grad, -self.grad_clip_value, self.grad_clip_value))

    def forward(self, input):
        x, h = input

        h_1, h_1_trans = self.rnn_1(self.fc_x_h_1(x))
        h_2, h_2_trans = self.rnn_2(h_1 + self.fc_x_h_2(x))
        h_3, h_3_trans = self.rnn_3(h_2 + self.fc_x_h_3(x))

        y = self.fc_h_y_1(h_1) + self.fc_h_y_2(h_2) + self.fc_h_y_3(h_3) + self.fc_res(x)

        if self.tanh:
            y = T.tanh(y)

        hidden_concat = T.cat((h_1_trans, h_2_trans, h_3_trans), dim=1)

        return y, hidden_concat

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

if __name__ == "__main__":
    env = type('',(object,),{'obs_dim':12,'act_dim':6})()
    config = {"policy_lastlayer_tanh" : False, "policy_memory_dim" : 96, "policy_grad_clip_value" : 1.0}
    RNN_PG(env, config)