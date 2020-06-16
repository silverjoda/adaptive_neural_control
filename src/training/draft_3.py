import math
import random
import torch as T
from torch import nn
from torch.nn import functional as F
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

X = T.tensor(3., requires_grad=True)
Y = T.tensor(2.,  requires_grad=True)
Z = 2 * X * Y + Y ** 2
L = (Z - 15) ** 2
L2 = T.abs(X - 2)

x_grad, y_grad = T.autograd.grad(L, [X, Y])
print(x_grad, y_grad)

xx_grad = T.autograd.grad(L2, X)
print(xx_grad)

