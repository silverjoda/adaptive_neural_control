import math



def solution(pegs):

    # Get lower and upper bounds on all gears
    gear_lbs, gear_ubs = get_bounds(pegs)

    nom = 0
    denom = 1
    sol = 0

    while True:
        if sol > 2:
            nom += 1
        else:
            denom += 1

        if upper_bound_reached():
            return [-1, -1]

        sol = forward(nom, denom, pegs)
        if sol == 2:
            return simplify(nom, denom)


def get_bounds(pegs):
    N = len(pegs)

    # Check just in case they try pull a sneaky
    for i in range(N - 1):
        if pegs[i + 1] - pegs[i] < 2:
            return [-1, -1]

    # List of gear lb and ub
    gear_lbs = [1] * N
    gear_ubs = [1] * N

    # Search gears iteratively backwards to calculate ub and lb
    for i in reversed(range(N)):
        # Edge case for last peg (rightmost)
        if i == (N - 1):
            gear_lbs[i] = 1
            gear_ubs[i] = pegs[i] - pegs[i - 1] - 1

        dist_to_right_peg = pegs[i + 1] - pegs[i]
        gear_lbs[i] = dist_to_right_peg - gear_ubs[i + 1]
        gear_ubs[i] = dist_to_right_peg - gear_lbs[i + 1]

    return gear_lbs, gear_ubs

def upper_bound_reached():
    pass

def forward(nom, denom, pegs):
    pass

def simplify(nom, denom):
    return 0

def intify(nom, denom):
    return 0

def main():
    print(solution([4, 30, 50]))
    print(solution([4, 17, 50]))

if __name__=="__main__":
    main()