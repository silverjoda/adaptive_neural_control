import torch as T
import torch.nn as nn

x_dim, y_dim, hid_dim, num_layers = 3, 1, 8, 1

rnn = nn.RNN(input_size=x_dim,hidden_size=hid_dim,
             num_layers=num_layers,batch_first=True)

rnn_cell = nn.RNNCell(input_size=x_dim,
                      hidden_size=hid_dim)

fc = nn.Linear(hid_dim, y_dim)

batchsize = 24
seq_len = 10

# All at once
X = T.randn(batchsize, seq_len, x_dim, requires_grad=True)
H, _ = rnn(X)
Y = fc(H)

# Step by step with Cell
h = None
for i in range(seq_len):
    h = rnn_cell(X[:,i,:], h)
    y = fc(h)






