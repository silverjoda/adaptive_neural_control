import torch.nn as nn
import torch.nn.functional as F
import torch as T
import numpy as np
import yaml
import math

# Phase 1: Learn on a single model (no rnd) with reactive and RNN nets.
# See if we can identify difficult states (try ensembles, dropout, state counting)
# In this phase we assume the data to be contiguous, one single long episode (no env randomization)

class DataLoader:
    def __init__(self, config):
        self.config = config
        path = config["data_path"]

        self.data_action = np.load(path + "action")
        self.data_angular_vel = np.load(path + "angular_vel")
        self.data_position = np.load(path + "position")
        self.data_rotation = np.load(path + "rotation")
        self.data_timestamp = np.load(path + "timestamp")
        self.data_vel = np.load(path + "vel")

        self.N_data_points = len(self.data_action)

        assert len(self.data_action) \
               == len(self.data_angular_vel) \
               == len(self.data_position) \
               == len(self.data_rotation) \
               == len(self.data_timestamp) \
               == len(self.data_vel)

        self.split_data()

    def split_data(self):
        self.n_chunks = math.floor(self.N_data_points / self.config["chunk_size"])

    def get_batch(self, N):
        pass

class Trainer:
    def __init__(self, config, dataloader, model):
        self.config = config
        self.dataloader = dataloader
        self.model = model

    def train(self):
        pass

    def eval(self):
        pass

class ReactiveModel(nn.Module):
    def __init__(self, config):
        super(ReactiveModel, self).__init__()
        self.config = config

        self.fc1 = nn.Linear(config["in_dim"], config["hid_dim"])
        self.fc2 = nn.Linear(config["hid_dim"], config["hid_dim"])
        self.fc3 = nn.Linear(config["hid_dim"], config["act_dim"])

        if self.config["policy_residual_connection"]:
            self.fc_res = nn.Linear(self.obs_dim, self.act_dim)

        self.activation_fun = eval(config["activation_fun"])

        T.nn.init.zeros_(self.fc1.bias)
        T.nn.init.zeros_(self.fc2.bias)
        T.nn.init.zeros_(self.fc3.bias)
        T.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        T.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        T.nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='linear')

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

class RnnModel(nn.Module):
    def __init__(self, config):
        super(RnnModel, self).__init__()
        self.config = config

        self.fc_in = nn.Linear(config["in_dim"], config["memory_dim"])
        self.rnn = nn.LSTM(config["memory_dim"], config["memory_dim"], 2, batch_first=True)
        self.fc_out = nn.Linear(config["memory_dim"], config["out_dim"])
        self.fc_res = nn.Linear(config["in_dim"], config["out_dim"])

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

def main():
    with open("configs/default.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    dataloader = DataLoader(config)
    model = ReactiveModel(config)
    trainer = Trainer(config, dataloader, model)
    trainer.train()
    trainer.eval()

if __name__ == "__main__":
    main()