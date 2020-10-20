import torch as T

x = T.tensor(1e12, requires_grad=True)
w = T.tensor(1e-12, requires_grad=True)
f = T.sigmoid(x*w)
f.backward()
print(x.grad, w.grad)