import torch as T
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

figure, axes = plt.subplots()
N = 30
fib = lambda n:reduce(lambda x,n:[x[1],x[0]+x[1]], range(n),[0,1])[0]
#t = [fib(i+2) / 20 for i in range(N)]
#t = np.linspace(0, 10, N)
t = [i**np.pi / 100 for i in range(N)]
print(t)

for i in t:
    draw_circle = plt.Circle((0.5, i), i, fill=False)
    plt.gcf().gca().add_artist(draw_circle)

plt.title('Circle')
axes.set_aspect(1)
plt.xlim(-10,10)
plt.ylim(0,20)
plt.show()