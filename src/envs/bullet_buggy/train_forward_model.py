import numpy as np
import torch as T
import torch.nn as nn
import torch.functional as F

class ForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.l1 = nn.Linear(self.config["input_dim"], self.config["hidden_dim"])
        self.l2 = nn.Linear(self.config["hidden_dim"], self.config["hidden_dim"])
        self.l3 = nn.Linear(self.config["hidden_dim"], self.config["output_dim"])

        self.non_linearity = eval(self.config["non_linearity"])

    def forward(self, x):
        feat1 = self.non_linearity(self.l1(x))
        feat2 = self.non_linearity(self.l2(feat1))
        out = self.l1(feat2)
        return out

    def predict(self, obs, act):
        x = T.tensor(np.concatenate((obs, act)), dtype=T.float32).unsqueeze(0)
        return self.forward(x)

    def predict_batch(self, obs, act):
        x = T.tensor(np.concatenate((obs, act), axis=1), dtype=T.float32)
        return self.forward(x)