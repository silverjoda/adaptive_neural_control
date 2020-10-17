import math
import numpy as np
from fractions import Fraction
import time

def solution(pegs):
    N = len(pegs)

    # Check if even or odd, this affects the output ratio differential w.r.t input
    even_n_pegs = N % 2 == 0

    # Get lower and upper bounds on all gears.
    gear_lbs, gear_ubs = get_bounds(pegs)
    if (np.array(gear_lbs) < 1).any():
        return [-1, -1]

    # Initialize variables.
    solution = [gear_lbs[0], 1]

    # Go!
    while True:
        final_ratio = forward(Fraction(*solution), pegs, gear_lbs, gear_ubs)

        # Found solution
        if final_ratio == 2:
            solution_simplified = Fraction(*solution)
            return [solution_simplified.numerator, solution_simplified.denominator]

        if ((final_ratio > 2) != even_n_pegs):
            # Increment numerator by 1.
            solution[0] += 1
        else:
            # If we are already at 1 then nowhere to go
            if Fraction(*solution) <= gear_lbs[0]:
                return [-1, -1]

            # Increment denominator by 1.
            solution[1] += 1

        # If we are searching beyond the bound of the first gear then there is no solution.
        if Fraction(*solution) > gear_ubs[0] + 1:
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
        gear_lbs[i] = max(dist_to_right_peg - gear_ubs[i + 1], 1)
        gear_ubs[i] = dist_to_right_peg - gear_lbs[i + 1]

    return gear_lbs, gear_ubs

def forward(solution, pegs, gear_lbs, gear_ubs):
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

        # Prune
        if not(gear_lbs[i] <= current_gear_size <= gear_ubs[i]):
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

    # for c in range(2, 100):
    #     for a in range(c+1, 100):
    #         for b in range(c+1, 100):
    #             for k in range(1, 100):
    #                 if a + b == c * k and (a / c) == (b / c) * 2 and a % c != 0 and b % c != 0:
    #                     print(a,b,c,k, [1, 1 + (a + b) / c])
    #
    #
    # exit()
    rnd_seed = 6653151 # int((time.time() % 1) * 10000000)
    np.random.seed(rnd_seed)
    # print(rnd_seed)
    # pegs = [1, 18, 20] # [15, 31, 42, 55, 66] should have a fraction output maybe
    # print(f"solution: {solution(pegs)}")
    # for sol in np.linspace(1, 20, 20):
    #     print(sol, forward_test(sol, pegs))

    N = 6
    for i in range(100):
        print(i)
        pegs = np.cumsum(np.random.randint(2, 50, size=N))
        sol = solution(pegs)

        if sol == [-1, -1] and False:
            print("---------------")
            print("---------------")
            print(f"For Pegs: {pegs}")
            clean = False
            for sol in np.linspace(1, 50, 100):
                result = forward_test(sol, pegs)
                print(sol, result)
                if result == -1 and clean: break
                if result != -1:
                    clean = True
            print("===============")
            print("===============")

        print(i, pegs, sol)

    #main()