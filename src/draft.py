import torch as T

x1 = T.tensor([.9], requires_grad=True)
x2 = T.tensor([.9], requires_grad=True)
a1 = T.log(x1)
a2 = x2

a1.backward()
a2.backward()

print(x1.grad)
print((1/x2) * x2.grad)