import torch as T
import torch.nn as nn

x_dim = 3
y_dim = 1
hid_dim = 8
num_layers = 1

rnn = nn.RNN(input_size=x_dim,
             hidden_size=hid_dim,
             num_layers=num_layers,
             batch_first=True)

fc = nn.Linear(hid_dim, y_dim)

batchsize = 24
seq_len = 10
X = T.randn(batchsize, seq_len, x_dim,
            requires_grad=True)
H, _ = rnn(X)
Y = fc(H)



