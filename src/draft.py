import torch as T
N = 5
X = T.randn(N,N, requires_grad=True)
M = T.randn(N,N, requires_grad=True)
lr = 0.01

for i in range(1000):
    X1 = M @ X
    xtx = X1.T @ X1
    loss = T.mean(xtx[0,0] + xtx[1,1] + xtx[2,2] + xtx[3,3] + xtx[4, 4])
    loss.backward()
    M = M - M.grad * lr
    if i % 10 == 0:
        print(i, loss)


