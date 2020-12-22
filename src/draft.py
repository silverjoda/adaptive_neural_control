import torch as T

a = T.tensor([10.], requires_grad=True)
# b = T.tensor([0.], requires_grad=True)
# c = T.tensor([1.0], requires_grad=True)
# l = T.tensor([-1.], requires_grad=True)
# v = a + b
# y = c * v



y = (a * 0.5) * (a * 3)
y.backward()

print(a.grad)
