import torch as T
import torch.nn as nn

# 2d
T.manual_seed(2)
f = nn.Conv2d(2,1,2,1,0,1,bias=True)
f.weight.data = T.randint(low=-5, high=5, size=(1,2,2,2), dtype=T.float32)
f.bias.data = T.randint(low=-5, high=5, size=[1], dtype=T.float32)
X = T.randint(low=-5, high=5, size=(1,2,2,2), dtype=T.float32)
Y = f(X)
print(X)
print(f.weight)
print(f.bias)
print(Y)

# 3d
T.manual_seed(2)
f = nn.Conv3d(1,1,2,1,0,1,bias=True)
f.weight.data = T.randint(low=-5, high=5, size=(1,1,2,2,2), dtype=T.float32)
f.bias.data = T.randint(low=-5, high=5, size=[1], dtype=T.float32)
X = T.randint(low=-5, high=5, size=(1,1,2,2,2), dtype=T.float32)
Y = f(X)
print(X)
print(f.weight)
print(f.bias)
print(Y)





