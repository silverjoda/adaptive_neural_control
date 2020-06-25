import math
def p(t, d):
    p = []
    for k in range(d//2):
        w_k = 1 / math.pow(1e5, 2 * k / d)
        p.append(math.sin(w_k * t))
        p.append(math.cos(w_k * t))
    return p

print([p(t, 2) for t in range(30)])