import torch as T
import torch.nn as nn
import copy
from copy import deepcopy
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        l1 = self.fc1(x)
        l2 = self.fc2(l1)
        return l2

class CNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Conv2d(4, 16, 3)

    def forward(self, x):
        l1 = self.fc1(x)
        return l1

n1 = Net()
n2 = Net()

cn1 = CNet()
cX = T.rand((1, 4, 100, 100))
cY = cn1(cX)

n1.fc1.weight = nn.Parameter(n2.fc1.weight.clone())

for p1, p2 in zip(n1.parameters(), n2.parameters()):
    print(id(p1),id(p2))

X = T.rand(4, requires_grad=True)
X2 = T.tensor(X)
Y = n1(X)
Y2 = n2(X2)

Y2.mean().backward()
Y.mean().backward()

pass


