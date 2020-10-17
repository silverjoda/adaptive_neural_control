import math
from fractions import Fraction

def solution(pegs):

    # Get lower and upper bounds on all gears.
    gear_lbs, gear_ubs = get_bounds(pegs)

    # Initialize variables.
    solution = Fraction(gear_lbs[0])

    # Go!
    while True:
        final_ratio = forward(solution, pegs, gear_lbs, gear_ubs)

        if final_ratio < 1:
            return [-1, -1]

        if final_ratio == 2:
            return solution.numerator, solution.denominator

        if final_ratio > 2:
            # Increment numerator by 1.
            solution += Fraction(1, solution.denominator)
        else:
            # Increment denominator by 1.
            solution -= Fraction(solution.numerator, solution.denominator ** 2 + solution.denominator) # <- Lol.

        # If we are searching beyond the bound of the first gear then there is no solution.
        if solution > gear_ubs[0]:
           return [-1, -1]

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
            continue

        dist_to_right_peg = pegs[i + 1] - pegs[i]
        gear_lbs[i] = dist_to_right_peg - gear_ubs[i + 1]
        gear_ubs[i] = dist_to_right_peg - gear_lbs[i + 1]

    return gear_lbs, gear_ubs

def forward(solution, pegs, gear_lbs, gear_ubs):
    N = len(pegs)
    current_gear_size = solution
    current_ratio = 1

    for i in range(1, N):
        # Calculate mandatory size of next gear
        size_of_gear_to_the_right = pegs[i] - pegs[i - 1] - current_gear_size

        # Calculate the resulting ratio
        current_ratio = current_ratio * (current_gear_size / size_of_gear_to_the_right)

        current_gear_size = size_of_gear_to_the_right

        # Invalid solution
        if size_of_gear_to_the_right < 1:
            return -1

        # Prune
        if current_gear_size < gear_lbs[i] or current_gear_size > gear_ubs[i]:
            return -1

    return current_ratio

def forward_test(solution, pegs):
    N = len(pegs)
    current_gear_size = solution
    current_ratio = 1

    for i in range(1, N):
        # Calculate mandatory size of next gear
        size_of_gear_to_the_right = pegs[i] - pegs[i - 1] - current_gear_size

        # Invalid solution
        if size_of_gear_to_the_right < 1:
            return -1

        # Calculate the resulting ratio
        current_ratio = current_ratio * (current_gear_size / size_of_gear_to_the_right)

        current_gear_size = size_of_gear_to_the_right

    return current_ratio

def main():
    print(solution([4, 30, 50]))
    print(solution([4, 17, 50]))

if __name__=="__main__":
    # import numpy as np
    # N = 3
    # pegs = [4, 30, 50]#np.cumsum(np.random.randint(1,100,size=N))
    # print(pegs)
    # for i in np.linspace(1, 30, 100):
    #     print(f"Solution: {i}: ratio: {forward_test(i, pegs)}")
    # exit()

    # a = Fraction(3, 4)
    # a -= Fraction(a.numerator, a.denominator ** 2 + a.denominator)
    #
    # print(a)
    # exit()
    main()