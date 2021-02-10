import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np
from torch.nn.utils import weight_norm
from torch.distributions.beta import Beta

class FF_HEX_EEF(nn.Module):
    def __init__(self, obs_dim, act_dim, config):
        super(FF_HEX_EEF, self).__init__()
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # [tensor(0.7135), tensor(-0.6724), tensor(-0.8629), tensor(-1.0894), tensor(0.0725), tensor(3.4730), tensor(0.3511), tensor(0.4637), tensor(-3.4840), tensor(-2.8000), tensor(-0.4658)]
        # x_mult, y_offset, z_mult, z_offset, phase_offset_l, phase_offset_r, *phases
        self.learned_params = nn.ParameterList([nn.Parameter(T.tensor(0.0)) for _ in range(self.act_dim)])


    def forward(self, x):
        clipped_params = [np.clip(np.tanh(self.learned_params[0].data) * 0.075 * 0.5 + 0.075, 0.075, 0.15), # x_mult
                          np.clip(np.tanh(self.learned_params[1].data) * 0.085 * 0.5 + 0.085, 0.05, 0.12), # y_offset
                          np.clip(np.tanh(self.learned_params[2].data) * 0.075 * 0.5 + 0.075, 0.075, 0.15), # z_mult
                          np.clip(np.tanh(self.learned_params[3].data) * 0.1 * 0.5 + 0.1, 0.05, 0.15), # z_offset
                          self.learned_params[4].data, # phase_offset_l
                          self.learned_params[5].data, # phase_offset_r
                          self.learned_params[6].data,
                          self.learned_params[7].data,
                          self.learned_params[8].data,
                          self.learned_params[9].data,
                          self.learned_params[10].data,
                          self.learned_params[11].data]
        act = [param.data for param in clipped_params]
        return act

    def sample_action(self, _):
        with T.no_grad():
            act = self.forward(None)
        return act

class FF_HEX_JOINT_PHASES(nn.Module):
    def __init__(self, obs_dim, act_dim, config):
        super(FF_HEX_JOINT_PHASES, self).__init__()
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.learned_params = nn.ParameterList([nn.Parameter(T.tensor(0.0)) for _ in range(self.act_dim)])

    def forward(self, x):
        clipped_params = [np.tanh(self.learned_params[0].data) * 0,
                          np.tanh(self.learned_params[1].data) * 0.25 + 0.25,
                          np.tanh(self.learned_params[2].data) * 0.5,
                          np.tanh(self.learned_params[3].data) * 0.75 + 0.75,
                          np.tanh(self.learned_params[4].data) * 0.5,
                          np.tanh(self.learned_params[5].data) * 0.75 + 0.75,
                          *[self.learned_params[i].data for i in range(6, 24)],
                          ]
        act = [param.item() for param in clipped_params]
        return act

    def sample_action(self, _):
        with T.no_grad():
            act = self.forward(None)
        return act

class VF_AC(nn.Module):
    def __init__(self, obs_dim, config):
        super(VF_AC, self).__init__()
        self.config = config
        self.obs_dim = obs_dim
        self.hid_dim = config["vf_hid_dim"]

        if self.config["policy_residual_connection"]:
            self.fc_res = nn.Linear(self.obs_dim, self.act_dim)

        self.activation_fun = eval(config["activation_fun"])

        self.fc1 = nn.Linear(self.obs_dim, self.hid_dim)
        self.fc2 = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc3 = nn.Linear(self.hid_dim, 1)

        w_i_b = self.config["weight_init_bnd"]
        #nn.init.uniform_(self.fc3.bias, -w_i_b, w_i_b)
        #nn.init.uniform_(self.fc3.weight, -w_i_b, w_i_b)

        for p in self.parameters():
            p.register_hook(lambda grad: T.clamp(grad, -config["policy_grad_clip_value"], config["policy_grad_clip_value"]))

    def forward(self, x):
        l1 = self.activation_fun(self.fc1(x))
        l2 = self.activation_fun(self.fc2(l1))
        if self.config["policy_residual_connection"]:
            out = self.fc3(l2) + self.fc_res(x)
        else:
            out = self.fc3(l2)
        return out

    def get_value(self, x):
        return self.forward(T.tensor(x).unsqueeze(0)).squeeze(0).detach().numpy()

class PI_AC(nn.Module):
    def __init__(self, obs_dim, act_dim, config):
        super(PI_AC, self).__init__()
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.hid_dim = config["policy_hid_dim"]

        if self.config["policy_residual_connection"]:
            self.fc_res = nn.Linear(self.obs_dim, self.act_dim)

        self.activation_fun = eval(config["activation_fun"])
        self.log_std = nn.Parameter(T.ones(1, self.act_dim, requires_grad=False) * config["init_log_std_value"])

        self.fc1 = nn.Linear(self.obs_dim, self.hid_dim)
        self.fc2 = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc3 = nn.Linear(self.hid_dim, self.act_dim)

        w_i_b = self.config["weight_init_bnd"]
        #nn.init.uniform_(self.fc1.bias, -w_i_b, w_i_b)
        #nn.init.uniform_(self.fc2.bias, -w_i_b, w_i_b)
        nn.init.uniform_(self.fc3.bias, -w_i_b * 0.1, w_i_b * 0.1)
        #nn.init.uniform_(self.fc1.weight, -w_i_b, w_i_b)
        #nn.init.uniform_(self.fc2.weight, -w_i_b, w_i_b)
        nn.init.uniform_(self.fc3.weight, -w_i_b * 0.1, w_i_b * 0.1)

        for p in self.parameters():
            p.register_hook(lambda grad: T.clamp(grad, -config["policy_grad_clip_value"], config["policy_grad_clip_value"]))

    def forward(self, x):
        l1 = self.activation_fun(self.fc1(x))
        l2 = self.activation_fun(self.fc2(l1))
        if self.config["policy_residual_connection"]:
            out = self.fc3(l2) + self.fc_res(x)
        else:
            out = self.fc3(l2)
        return out

    def sample_action(self, s, deterministic=False):
        with T.no_grad():
            s_T = T.tensor(s).unsqueeze(0)
            act = self.forward(s_T)
            if deterministic:
                return act.detach().squeeze(0).numpy()
            rnd_act = T.normal(act, T.exp(self.log_std))
        return rnd_act.detach().squeeze(0).numpy()

    def sample_par_action(self, s):
        with T.no_grad():
            s_T = T.tensor(s)
            act = self.forward(s_T)
            rnd_act = T.normal(act, T.exp(self.log_std.expand_as(act)))
        return rnd_act.detach().numpy()

    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        var = T.exp(log_std_batch).pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)

class PI_AC_BETA(nn.Module):
    def __init__(self, obs_dim, act_dim, config):
        super(PI_AC_BETA, self).__init__()
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.hid_dim = config["policy_hid_dim"]

        if self.config["policy_residual_connection"]:
            self.fc_res = nn.Linear(self.obs_dim, self.act_dim)

        self.activation_fun = eval(config["activation_fun"])

        self.fc1 = nn.Linear(self.obs_dim, self.hid_dim)
        self.fc2 = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc3 = nn.Linear(self.hid_dim, self.act_dim * 2)

        #w_i_b = self.config["weight_init_bnd"]
        nn.init.uniform_(self.fc3.bias, -0.01, 0.01)
        nn.init.uniform_(self.fc3.weight, -0.01, 0.01)

        for p in self.parameters():
                p.register_hook(lambda grad: T.clamp(grad, -config["policy_grad_clip_value"], config["policy_grad_clip_value"]))

    def forward(self, x):
        l1 = self.activation_fun(self.fc1(x))
        l2 = self.activation_fun(self.fc2(l1))
        if self.config["policy_residual_connection"]:
            out = self.fc3(l2) + self.fc_res(x)
        else:
            out = self.fc3(l2)
        return out

    def sample_action(self, s):
        s_T = T.tensor(s).unsqueeze(0)
        act = self.forward(s_T)
        c1 = F.sigmoid(act[:, :self.act_dim]) * 5
        c2 = F.sigmoid(act[:, self.act_dim:]) * 5
        beta_dist = Beta(c1, c2)
        rnd_act = beta_dist.sample()
        return rnd_act.detach().squeeze(0).numpy()

    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        act = self.forward(batch_states)

        # Calculate probabilities
        c1 = F.sigmoid(act[:, :, self.act_dim]) * 5
        c2 = F.sigmoid(act[:, :, self.act_dim:]) * 5

        beta_dist = Beta(c1, c2)
        log_probs = beta_dist.log_prob(batch_actions)
        return log_probs.sum(1, keepdim=True)

class SLP_PG(nn.Module):
    def __init__(self, obs_dim, act_dim, config):
        super(SLP_PG, self).__init__()
        self.config = config

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.fc1 = nn.Linear(self.obs_dim, self.act_dim)
        self.log_std = T.zeros(1, self.act_dim)

        self.fc1.weight.data.uniform_(-0.3, 0.3)
        self.fc1.bias.data.uniform_(-0.3, 0.3)
        #nn.init.xavier_uniform_(self.fc1.weight.data)

    def forward(self, x):
        x_T = T.tensor(x).unsqueeze(0)
        x = self.fc1(x_T)
        x_np = x.squeeze(0).detach().numpy()
        return x_np

    def sample_action(self, s):
        act = self.forward(s)
        return act, T.normal(act, T.exp(self.log_std))

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

        self.policy_memory_dim = config["policy_memory_dim"]
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
            p.register_hook(lambda grad: T.clamp(grad, -config["policy_grad_clip_value"], config["policy_grad_clip_value"]))

    def forward(self, input):
        x, h = input
        rnn_features = self.activation_fun(self.fc1(x))

        rnn_output, h = self.rnn(rnn_features, h)

        y = self.fc_out(F.tanh(self.fc_res(x)) + rnn_output)
        if self.config["policy_lastlayer_tanh"]:
            y = T.tanh(y)

        return y, h

    def sample_action(self, s):
        x, h = self.forward(s)
        return x[0], T.normal(x[0], T.exp(self.log_std_cpu)), h

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
        return x[0], T.normal(x[0], T.exp(self.log_std_cpu)), h

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

class CYC_QUAD(nn.Module):
    def __init__(self, env, config):
        super(CYC_QUAD, self).__init__()

        self.phase_stepsize = 0.1
        self.phase_global = 0

        self.phase_scale_global = T.nn.Parameter(T.ones(1))
        self.phase_offset_joints = T.nn.Parameter(T.zeros(12))

    def forward(self, _):
        act = T.sin(self.phase_global + self.phase_offset_joints).unsqueeze(0)
        self.phase_global = (self.phase_global + self.phase_stepsize * self.phase_scale_global) % (2 * np.pi)
        return act

class CYC_HEX(nn.Module):
    def __init__(self, env, config):
        super(CYC_HEX, self).__init__()

        self.phase_stepsize = 0.1
        self.phase_global = 0

        self.phase_scale_global = T.nn.Parameter(T.ones(1))
        self.phase_offset_joints = T.nn.Parameter(T.zeros(18))

    def forward(self, _):
        act = T.sin(self.phase_global + self.phase_offset_joints).unsqueeze(0)
        self.phase_global = (self.phase_global + self.phase_stepsize * self.phase_scale_global) % (2 * np.pi)
        return act

class NN_ATTENTION(nn.Module):
    def __init__(self, env, config):
        super(NN_ATTENTION, self).__init__()
        self.env = env
        self.config = config
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.qk_dim = config["policy_qk_dim"]
        self.value_dim = config["policy_value_dim"]
        self.n_heads = config["policy_n_heads"]

        self.activation_fun = eval(config["activation_fun"])
        self.log_std = -T.ones(1, self.act_dim) * 0.0

        self.fc_emb = nn.Linear(self.obs_dim, self.hid_dim)
        self.fc_q = nn.Linear(self.hid_dim, self.hid_dim * self.n_heads)
        self.fc_k = nn.Linear(self.hid_dim, self.hid_dim * self.n_heads)
        self.fc_k = nn.Linear(self.hid_dim, self.act_dim * self.n_heads)
        self.fc_out = nn.Linear(self.act_dim * self.n_heads, self.act_dim)

    def forward(self, x):
        # x is [len x b x dim]
        seq_len = x.shape[0]
        queries = []
        keys = []
        values = []

        # Calculate q,k,v for all states
        q,k,v = self.calc_qkv_single(x.view(x.shape[0] + x.shape[1], x.shape[2]))

        attended_dotprods = T.matmul(q,k)
        attended_weights = T.softmax(attended_dotprods, dim=1)
        attended_outputs = T.matmul(attended_weights, v)

        return attended_outputs

    def forward_single(self, qkv, x):
        q,k,v = qkv
        q_x,k_x,v_x = self.calc_qkv_single(x)
        # Attend
        attended_dotprods = T.matmul(q_x, k)
        attended_weights = T.softmax(attended_dotprods, dim=1)
        attended_outputs = T.matmul(attended_weights, v)

    def calc_qkv_single(self, x):
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)
        return q,k,v

    def sample_action(self, s):
        act = self.forward(s)
        rnd_act = T.normal(act, T.exp(self.log_std))
        rnd_act_center = T.normal(T.zeros_like(act), T.exp(self.log_std)) + act
        return act, rnd_act

    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)

class NN_TEMPCONV(nn.Module):
    def __init__(self, env, config):
        super(NN_TEMPCONV, self).__init__()
        self.env = env
        self.config = config

        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.tempconvnet = TemporalConvNet(self.obs_dim, self.config["num_channels"])

    def forward(self, x):
        return self.tempconvnet(x)

    def sample_action(self, s):
        act = self.forward(s)
        rnd_act = T.normal(act, T.exp(self.log_std))
        rnd_act_center = T.normal(T.zeros_like(act), T.exp(self.log_std)) + act
        return act, rnd_act

    def sample_action_w_activations(self, l0):
        l1_activ = self.fc1(l0)
        l1_normed = self.m1(l1_activ)
        l1_nonlin = self.activaton_fun(l1_normed)

        l2_activ = self.fc2(l1_nonlin)
        l2_normed = self.m1(l2_activ)
        l2_nonlin = self.activaton_fun(l2_normed)

        l3 = self.fc3(l2_nonlin)
        return l1_activ.squeeze(0).detach().numpy(),\
               l1_normed.squeeze(0).detach().numpy(),\
               l1_nonlin.squeeze(0).detach().numpy(), \
               l2_activ.squeeze(0).detach().numpy(), \
               l2_normed.squeeze(0).detach().numpy(), \
               l2_nonlin.squeeze(0).detach().numpy(), \
               l3.squeeze(0).detach().numpy(), \
               T.normal(l3, T.exp(self.log_std)).squeeze(0).detach().numpy()

    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MLP_ES(nn.Module):
    def __init__(self, obs_dim, act_dim, config):
        super(MLP_ES, self).__init__()
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.hid_dim = config["policy_hid_dim"]

        if self.config["policy_residual_connection"]:
            self.fc_res = nn.Linear(self.obs_dim, self.act_dim)

        self.activation_fun = eval(config["activation_fun"])

        self.hidden_layers_list = nn.ModuleList()
        for i in range(config["n_hidden_layers"]):
            if i == 0:
                self.hidden_layers_list.append(nn.Linear(self.obs_dim, self.hid_dim))
            else:
                self.hidden_layers_list.append(nn.Linear(self.hid_dim, self.hid_dim))

        if config["n_hidden_layers"] > 0:
            self.final_layer = nn.Linear(self.hid_dim, self.act_dim)
        else:
            self.final_layer = nn.Linear(self.obs_dim, self.act_dim)

    def forward(self, x):
        l_hidden = x
        if self.config["n_hidden_layers"] > 0:
            for i in range(self.config["n_hidden_layers"]):
                l_hidden = self.activation_fun(self.hidden_layers_list[i](l_hidden))
        l_final = self.final_layer(l_hidden)

        if self.config["policy_residual_connection"]:
            out = l_final + self.fc_res(x)
        else:
            out = l_final
        return out

    def sample_action(self, s):
        with T.no_grad():
            s_T = T.tensor(s).unsqueeze(0)
            act = self.forward(s_T)
        return act.detach().squeeze(0).numpy()


if __name__ == "__main__":
    env = type('',(object,),{'obs_dim':12,'act_dim':6})()
    config_rnn = {"policy_lastlayer_tanh" : False, "policy_memory_dim" : 96, "policy_grad_clip_value" : 1.0}
    config_nn = {"policy_lastlayer_tanh": False, "policy_hid_dim": 96, "policy_grad_clip_value": 1.0}
    RNN_PG(env, config_rnn)