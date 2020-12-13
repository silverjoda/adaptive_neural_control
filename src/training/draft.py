import torch as T
from torch.distributions.beta import Beta

x = T.tensor([2., 2.], requires_grad=True)
m = Beta(x[0],x[1])
s = m.sample()
p = m.log_prob(s)
p.backward()

print(f"Sample s: {s}, log_prob: {p}, grad_x: {x.grad}")


