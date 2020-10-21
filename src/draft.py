
def sort_l(num, asc=True):
    pass

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

def l_to_num(l):
    pass

def num_to_l(n):
    pass

def sub_l(x, y, b):
    assert len(x) == len(y)
    res = [0] * len(x)
    for i in reversed(range(len(x))):
        if x[i] > y[i]:
            res[i] = x[i] - y[i]
        else:
            # x will always be larger than y
            res[i] = b - y[i] + x[i]
            if i > 0:
                res[i-1] =- 1

def check_repeat(ID, l):
    try:
        idx = l.index(ID)
    except:
        return -1
    return len(l) - idx

def solution(n, b):
    # Make n into string
    #n_str = [int(l) for l in n]
    cur_minion_ID = num_to_l(n)

    # Start and log initial minion ID (given)
    minion_ID_list = [cur_minion_ID]

    ## Start cycle:
    while True:
        # Check if z is in the log
        cycle_len = check_repeat(cur_minion_ID, minion_ID_list)
        if cycle_len > -1:
            return cycle_len

        minion_ID_list.append(cur_minion_ID)

        # Turn minion number repre into list
        cur_minion_ID_l = num_to_l(cur_minion_ID)

        # Sort nums to get x,y
        x = sort_l(cur_minion_ID_l, asc=True)
        y = sort_l(cur_minion_ID_l, asc=False)

        # Subtract nums (z is padded from the left)
        z = sub_l(x, y, b)

        cur_minion_ID = l_to_num(z)


def main():
    pass

if __name__=="__main__":
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