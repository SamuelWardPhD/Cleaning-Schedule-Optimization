import itertools
import numpy as np
import pandas as pd
from typing import Callable, Sequence

"""
This code was designed in conjunction with the following paper and should be cited as:
    Samuel Ward, Marah-Lisanne Thormann, Julian Wharton and Alain Zemkoho (2026).
    Data-Driven Hull-Fouling Cleaning Schedule Optimization to Reduce Carbon Footprint of Vessels.


Optimisation Problem:
    minimise:       sum_{i=1,...,n} f(X[i], b[i]) + c[i]*z[i]
    subject to:     z[i] in {0, 1}
                    b[i] = accumulation of biofouling


Variables:    
    n     = Number of voyages.
    c[i]  = Cost of cleaning the ship before voyage i.
    X[i]  = Data profile of voyage i (e.g. speed, cargo, trim ...).
    B0    = Measure of the initial level of biofouling.
    B[i]  = Measure of the additional biofouling due to voyage i.
    b[i]  = Cumulative measure of the total biofouling up to voyage i that resets to zero when cleaned.
    f     = Regression function that predicts the cost of a voyage.
    z[i]  = The decision variable which takes value 1 if we decide to clean before voyage i and 0 otherwise.


File structure:
    example_problem_instance()
    algorithm_1_brute_force_search(n, b0, c, X, B, f)
    algorithm_2_dynamic_cleaning_schedule_optimiser(n, b0, c, X, B, f)
    main()
"""


def example_problem_instance():
    # Number of voyages
    n = 5

    # Initial level of biofouling (100 days)
    b0 = 100

    # Cost of cleaning at each port (USD)
    c = np.array((45_000, 44_000, 39_000, 36_000, 42_000))

    # Each voyage increases biofouling level by 25 days
    B = np.array((25, 25, 25, 25, 25))

    # Example DataFrame for each of the five voyages
    X = (
        # Voyage 1
        pd.DataFrame({
            "speed_through_water_kn": [8.5, 9.2, 10.0, 11.3, 12.1, 10.8, 9.7, 11.9, 13.0, 12.4],
            "draught_plus_wave_height_m": [9.8, 10.1, 10.0, 10.4, 10.6, 10.3, 10.2, 10.7, 10.9, 10.8]
        }),

        # Voyage 2
        pd.DataFrame({
            "speed_through_water_kn": [9.0, 9.6, 10.4, 11.0, 12.0, 12.3, 11.1, 12.7, 13.2, 12.8],
            "draught_plus_wave_height_m": [9.7, 9.9, 10.0, 10.1, 10.2, 10.1, 9.8, 10.3, 10.4, 10.2]
        }),

        # Voyage 3
        pd.DataFrame({
            "speed_through_water_kn": [8.2, 8.9, 9.5, 10.1, 10.7, 10.3, 9.6, 10.8, 11.5, 11.0],
            "draught_plus_wave_height_m": [10.0, 10.3, 10.5, 10.7, 10.8, 10.6, 10.4, 10.9, 11.0, 10.8]
        }),

        # Voyage 4
        pd.DataFrame({
            "speed_through_water_kn": [9.5, 10.2, 10.8, 11.6, 12.5, 12.0, 11.3, 12.9, 13.5, 13.0],
            "draught_plus_wave_height_m": [9.6, 9.8, 10.0, 10.1, 10.2, 10.0, 9.9, 10.3, 10.4, 10.2]
        }),

        # Voyage 5
        pd.DataFrame({
            "speed_through_water_kn": [8.8, 9.4, 10.1, 10.9, 11.7, 11.2, 10.5, 11.8, 12.6, 12.1],
            "draught_plus_wave_height_m": [9.9, 10.0, 10.2, 10.3, 10.5, 10.4, 10.1, 10.6, 10.7, 10.5]
        }),
    )

    # Regression function to predict voyage cost
    def f(_X, _b):
        fuel_cost_USD_per_kg = 0.493
        fuel_oil_consumption_kg = sum(
            _X['draught_plus_wave_height_m'][i]*(_X['speed_through_water_kn'][i]**2) + 36*_b
            for i in range(n)
        )
        return fuel_oil_consumption_kg*fuel_cost_USD_per_kg

    # Return a parameterization that can be passed to Algorithms 1 and 2
    return n, b0, c, X, B, f


def algorithm_1_brute_force_search(
        n: int,
        b0: float,
        c: Sequence,
        X: Sequence,
        B: Sequence,
        f: Callable,
        verbose=3,
):
    """
    :param n:          Number of voyages. Should be an integer greater than one;
    :param b0:         Initial value of the biofouling measure. Non-negative float;
    :param c:          Numpy array where c[i] is the cost of cleaning before voyage i;
    :param B:          Numpy array where B[i] is the measure of biofouling that accumulates for voyage i;
    :param X:          Array where X[i] is the profile of voyage i.  Each element may be a DataFrame or other object;
    :param f:          Function f which outputs the cost given a voyage profile;
    :param verbose:    Level of output. Set to zero to supress all print statements;
    """

    # Title
    if verbose:
        print("\n===== Brute Force =====")

    # The objective function of the integer program
    # sum_{i=1,...,n}( f(X[i], b[i]) + c[i]*z[i] )
    def objective(_z):
        b = np.zeros(n, dtype=float)
        count = b0
        for i in range(n):
            if _z[i] == 1:
                count = 0
            b[i] = count
            count += B[i]

        return sum((
            f(X[i], b[i]) for i in range(n)
        )) + sum((
            c[i]*_z[i] for i in range(n)
        ))

    # Initialise variables that will store the best solution found so far
    z_best = None
    obj_best = np.inf

    # Iterate through every combination of decision variables
    # I.e. (0, 0, ...) then (1, 0, ...) then (0, 1, ...) then (1, 1, ...) and so on
    for progress, z in enumerate(itertools.product((0, 1), repeat=n)):
        obj = objective(z)

        # Maintain the best solution so far
        if obj < obj_best:
            obj_best = obj
            z_best = z

        # Update
        if verbose >= 2 and (progress+1) in (((2**n)*p)//100 for p in range(20, 101, 20)):
            print(f"   {progress+1:>3} out of {2**n} combinations searched ({(progress+1)/(2**n):.1%})")

    # Output results to console
    if verbose:
        print(f"Optimal cleaning schedule z*:   {z_best}")
        print(f"Optimal object value ($cost):   {obj_best:.2f}")

    return z_best, obj_best


def algorithm_2_dynamic_cleaning_schedule_optimiser(
        n: int,
        b0: float,
        c: Sequence,
        X: Sequence,
        B: Sequence,
        f: Callable,
        b0_is_B0: bool = False,
        verbose=3,
):
    """
    :param n:          Number of voyages. Should be an integer greater than one;
    :param b0:         Initial value of the biofouling measure. Non-negative float;
    :param c:          Numpy array where c[i] is the cost of cleaning before voyage i;
    :param B:          Numpy array where B[i] is the measure of biofouling that accumulates for voyage i;
    :param X:          Array where X[i] is the profile of voyage i.  Each element may be a DataFrame or other object;
    :param f:          Function f which outputs the cost given a voyage profile;
    :param b0_is_B0:   Set this to true if B[0] = b0. Otherwise, we will set B = [b0] + B;
    :param verbose:    Level of output. Set to zero to supress all print statements;
    """

    # Validation
    assert len(c) >= n, f"Invalid parameterization: len(c)={len(c)} should be equal to n={n}."
    assert len(B) >= n, f"Invalid parameterization: len(B)={len(B)} should be equal to n={n}."
    assert len(X) >= n, f"Invalid parameterization: len(X)={len(X)} should be equal to n={n}."

    # Title
    if verbose:
        print("\n===== Dynamic Programming =====")

    # To understand the dynamic programing algorithm you must consider the following sub-problem
    # CLEANING[i,j] is the sub-problem of solving for the subset of variables ( z_i, ... , z_n ) given that z_j=1
    #
    # Phi[i,j] is the optimal objective value for CLEANING[i,j]
    # Psi[i,j] is the optimal cleaning schedule for CLEANING[i,j]
    Phi = np.zeros((n, n))
    Psi = [[tuple((0 for __ in range(i + 1))) for _ in range(n - i)] for i in reversed(range(n))]

    # We wish B[0]=b_0 to refer to the initial biofouling and B[1] to refer to the first voyage
    if not b0_is_B0:
        B = np.concatenate((np.array([b0]), B[:-1]))

    # Vector of zeros
    zero = 0 if len(B.shape) == 1 else np.zeros(B.shape[1])

    # Work backwards through the voyages
    for i in reversed(range(n)):

        if verbose >= 2:
            print(f"   --- Voyage {i + 1} ---")

        # The cost assuming we clean at voyage i is the sum of
        #   + The cost of cleaning C[i]
        #   + The fuel cost of the voyage with zero biofouling f(X[i], 0)
        #   + The optimal cost of future voyages' Phi[i + 1, i + 1]
        phi_clean = c[i] + f(X[i], zero) + (0 if i + 1 == n else Phi[i + 1, i + 1])
        z_clean = tuple((1,)) if i + 1 == n else (tuple((1,)) + Psi[i + 1][i + 1])

        # Assume that we last cleaned at before voyage
        for j in reversed(range(i + 1)):

            # The cost assuming we have not cleaned since voyage j
            #   + The fuel cost of the voyage with b biofouling f(X[i], b)
            #   + The optimal cost of future voyages' Phi[i + 1, j]
            b = np.sum(B[j:i + 1], axis=0)
            phi_fouled = f(X[i], b) + (0 if i + 1 == n else Phi[i + 1, j])
            z_fouled = tuple((0,)) if i + 1 == n else (tuple((0,)) + Psi[i + 1][j])

            if verbose >= 2:
                print("".join((
                    f"j: {j:<4}",
                    f"b: {np.sum(B[j:i + 1]):<6}",
                    f"phi_clean: {phi_clean:<10.2f}",
                    f"phi_fouled: {phi_fouled:<10.2f}",
                    f"B[{j}:i + 1]: {B[j:i + 1]}"
                )))

            # If it is optimal to clean, record so in tables Phi and Psi
            if phi_clean <= phi_fouled:
                Phi[i][j] = phi_clean
                Psi[i][j] = z_clean

            # If it is optimal to level the biofouling, record so in tables Phi and Psi
            else:
                Phi[i][j] = phi_fouled
                Psi[i][j] = z_fouled

    # Output results to console
    if verbose:
        print("   --- Results ---")
        print(f"Optimal cleaning schedule z*:   {Psi[0][0]}")
        print(f"Optimal object value ($cost):   {Phi[0][0]:.2f}")

    return Psi[0][0], Phi[0][0]


def main():
    # Get the parameters for the example problem instance
    n, b0, c, X, B, f = example_problem_instance()

    # Solve with brute force
    algorithm_1_brute_force_search(n, b0, c, X, B, f)

    # Solve with dynamic programing
    algorithm_2_dynamic_cleaning_schedule_optimiser(n, b0, c, X, B, f)


if __name__ == '__main__':
    main()

__author__ = "Samuel Ward"
__maintainer__ = "Samuel Ward"
__email__ = "s.ward@soton.ac.uk"
__credits__ = ["Samuel Ward", "Marah-Lisanne Thormann"]
__version__ = "7.0.2"
