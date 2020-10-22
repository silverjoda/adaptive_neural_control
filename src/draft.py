import math

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

def l_to_num(l, b):
    len_ = len(l)
    return int(sum([l[i] * math.pow(b, len_ - i - 1) for i in range(len_)]))

def sub_l_manual(x, y, b):
    assert len(x) == len(y)
    res = [0] * len(x)
    carry = 0
    for i in reversed(range(len(x))):
        if x[i] >= y[i]:
            res[i] = x[i] - y[i]
        else:
            res[i] = b + x[i] - y[i]
            if x[i-1] > 0:
                x[i - 1] = x[i - 1] - 1
            else:
                x[i - 1] = b - 1
                carry = 1

    return res

def num_to_l(n, b, len_):
    l = [0] * len_
    for i in range(len_):
        n_pos = math.pow(b, len_ - i - 1)
        l[i] = int(n / int(n_pos))
        n = n - n_pos * l[i]
    return l

def sub_l(x, y, b):
    x_int = int(''.join(str(n) for n in x), b)
    y_int = int(''.join(str(n) for n in y), b)
    z_int = x_int - y_int
    z_list = num_to_l(z_int, b, len(x))
    return z_list

def check_repeat(ID, l):
    if len(l) < 2:
        return -1
    try:
        idx = l.index(ID)
    except:
        return -1
    return len(l) - idx

def solution(n, b):
    # Make n into string
    cur_minion_ID = [int(l) for l in n]

    # Start and log initial minion ID (given)
    minion_ID_list = []

    ## Start cycle:
    while True:
        # Check if z is in the log
        cycle_len = check_repeat(cur_minion_ID, minion_ID_list)
        if cycle_len > -1:
            return cycle_len

        minion_ID_list.append(cur_minion_ID)

        # Sort nums to get x,y
        x = sorted(cur_minion_ID, reverse=True)
        y = sorted(cur_minion_ID, reverse=False)

        # Subtract nums (z is padded from the left)
        z = sub_l(x, y, b)

        cur_minion_ID = z

def main():
    print(solution('210022', 3))

if __name__=="__main__":
    #sub_l([1,6,0,5], [0,3,7,9], 10)
    main()
    exit()
    w = [1,5,2,44,44,0,11]
    try:
        idx = w.index(11)
    except:
        idx = -1

    cycle_len = len(w) - idx
    print(idx, cycle_len)
    exit()

    a = int('0101100', base=2)
    b = int('0001100', base=2)
    c = a + b
    print(a, ''.join(map(str, numberToBase(a, 10))))
    main()