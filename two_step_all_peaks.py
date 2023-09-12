# Two-step optimization where the first steps try to optimize the problem with ramp constraints
# and total energy generation constraint included. The second step includes the peak constraints to the optimization
# problem. This particular script considers all local peaks for the given schedule based on a peak finding criteria

import sys
import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

data = pd.read_csv('ramp_constraints.csv')

# Variable for the new generation schedule
G_new = cp.Variable(len(data['Generation Schedule']), nonneg=True)

# Original generation schedule
G_original = data['Generation Schedule'].values

objective_terms = cp.multiply(cp.abs(G_new - G_original), G_original ** 2)
objective = cp.Minimize(cp.sum(objective_terms))

# Ramp rate constraints
ramp_diffs = G_new[1:] - G_new[:-1]
ramp_constraints = [cp.abs(ramp_diffs) <= 450]

# ramp_constraints_explicit = []
# for hour in range(1, len(data)):  # Starting from 1 as we're comparing with the previous hour
#     ramp_constraints_explicit.append(G_new[hour] - G_new[hour - 1] <= 450)
#     ramp_constraints_explicit.append(G_new[hour - 1] - G_new[hour] <= 450)

# Total energy constraint
energy_constraint = [cp.sum(G_new) == cp.sum(G_original)]

# phase 1 only considers ramp constraints and energy constraints
constraints_phase1 = ramp_constraints + energy_constraint

problem_phase1 = cp.Problem(objective, constraints_phase1)
print(problem_phase1)
problem_phase1.solve()

if problem_phase1.status != 'optimal':
    sys.exit("Phase 1 optimization problem is infeasible with the given constraints")
else:
    print("Phase 1 optimal solution found")
    print("Proceed to Phase 2 optimization problem to include peak constraints")
    adjusted_generation_phase1 = G_new.value
    plt.figure(figsize=(15, 6))
    plt.plot(data['Hour'], data['Generation Schedule'], label='Original Generation', color='blue')
    plt.plot(data['Hour'], adjusted_generation_phase1, label='Adjusted Generation (Phase 1)', color='red',
             linestyle='--')
    plt.xlabel('Hour')
    plt.ylabel('Generation (MW)')
    plt.xticks(range(0, len(data["Hour"]) + 1, 24))  # setting x ticks at 24 hr intervals to indicate each day
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title('Original and Adjusted Generation Schedules (Phase 1)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots\\phase1_opt_sol.png")

# Phase 2
# Detect local peaks in the generation schedule
# Prepend and append a value lower than the first and last data points respectively
extended_schedule = np.insert(data['Generation Schedule'].values, 0, data['Generation Schedule'].iloc[0] - 1)
extended_schedule = np.append(extended_schedule, data['Generation Schedule'].iloc[-1] - 1)
local_peaks_indices, _ = find_peaks(extended_schedule)

# Adjust indices to match original data
local_peaks_indices = local_peaks_indices - 1

# Create constraints for these local peaks
local_peak_constraints = [G_new[idx] == G_original[idx] for idx in local_peaks_indices]

# Modify constraints for Phase 2 to use local peak constraints
constraints_phase2 = ramp_constraints + energy_constraint + \
                     local_peak_constraints

# Solve the Phase 2 problem with the modified constraints
problem_phase2 = cp.Problem(objective, constraints_phase2)
problem_phase2.solve()

if problem_phase2.status == 'optimal':
    print("Phase 2 optimal solution found")
    adjusted_generation_phase2 = G_new.value
    plt.figure(figsize=(15, 6))
    plt.plot(data['Hour'], data['Generation Schedule'], label='Original Generation', color='blue')
    plt.plot(data['Hour'], adjusted_generation_phase2, label='Adjusted Generation (Phase 2)', color='green',
             linestyle='--')
    plt.xlabel('Hour')
    plt.ylabel('Generation (MW)')
    plt.xticks(range(0, len(data["Hour"]) + 1, 24))  # setting x ticks at 24 hr intervals to indicate each day
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title('Original and Adjusted Generation Schedules (Phase 2)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots\\phase2_opt_sol.png")
    sys.exit()
else:
    print("Proceed to relax peak constraints and find the minimum tolerance")

upper_limit = 0.1
lower_limit = 0.001
epsilon = 0.001

# relax local peak constraints and find the minimum tolerance where the solution is feasible
while upper_limit - lower_limit > epsilon:
    tolerance = (upper_limit + lower_limit) / 2

    local_peak_constraints_relaxed_lower = [
        G_new[idx] >= (1 - tolerance) * G_original[idx]
        for idx in local_peaks_indices
    ]
    local_peak_constraints_relaxed_upper = [
        G_new[idx] <= (1 + tolerance) * G_original[idx]
        for idx in local_peaks_indices
    ]

    # Combine them into a single list for local peaks
    local_peak_constraints_relaxed = local_peak_constraints_relaxed_lower + local_peak_constraints_relaxed_upper

    # Modify constraints for Phase 2 to use local peak constraints
    constraints_phase2_tol = ramp_constraints + energy_constraint + local_peak_constraints_relaxed

    # Solve the Phase 2 problem with the modified constraints
    problem_phase2 = cp.Problem(objective, constraints_phase2_tol)
    problem_phase2.solve()

    if problem_phase2.status == 'optimal':
        upper_limit = tolerance
        adjusted_generation_phase2 = G_new.value
    else:
        lower_limit = tolerance

header = "Adjusted Generation"
np.savetxt('adjusted_generation.csv', adjusted_generation_phase2, delimiter=',', header=header, fmt='%d', comments='')
print("Phase 2 optimal solution found")
plt.figure(figsize=(15, 6))
plt.plot(data['Hour'], data['Generation Schedule'], label='Original Generation', color='blue')
plt.plot(data['Hour'], adjusted_generation_phase2, label='Adjusted Generation (Phase 2)', color='green',
         linestyle='--')
plt.xlabel('Hour')
plt.ylabel('Generation (MW)')
plt.title('Original and Adjusted Generation Schedules (Phase 2)')
plt.legend()
textstr = f'Value: {tolerance}'
props = dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5)
plt.annotate(textstr, xy=(0, 1), xycoords='axes fraction', fontsize=12,
             verticalalignment='top', bbox=props)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("plots\\phase2_opt_sol.png")

plt.figure(figsize=(15, 6))
plt.plot(data['Hour'], data['Generation Schedule'], label='Original Generation', color='blue')
plt.plot(data['Hour'], adjusted_generation_phase1, label='Adjusted Generation (Phase 1)', color='red',
         linestyle='--')
plt.plot(data['Hour'], adjusted_generation_phase2, label='Adjusted Generation (Phase 2)', color='green',
         linestyle='--')
plt.xlabel('Hour')
plt.ylabel('Generation (MW)')
plt.xticks(range(0, len(data["Hour"]) + 1, 24))  # setting x ticks at 24 hr intervals to indicate each day
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.title('Original and Adjusted Generation Schedules')
plt.legend()
props = dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5)
plt.annotate(textstr, xy=(0, 1), xycoords='axes fraction', fontsize=12,
             verticalalignment='top', bbox=props)
plt.tight_layout()
file_path = "plots\\opt_sol.png"
plt.savefig(file_path)
k = 1
