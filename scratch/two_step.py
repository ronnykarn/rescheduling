import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


data = pd.read_csv('../ramp_constraints.csv')
# Variable for the new generation schedule
G_new_phase1 = cp.Variable(len(data['Generation Schedule']))

# Original generation schedule
G_original = data['Generation Schedule'].values

# Objective for Phase 1 (just feasibility)
objective_phase1 = cp.Minimize(0)

# Constraints
ramp_diffs_phase1 = G_new_phase1[1:] - G_new_phase1[:-1]
ramp_constraints_phase1 = [cp.abs(ramp_diffs_phase1) <= 450]
non_negativity_constraint_phase1 = [G_new_phase1 >= 0]
energy_constraint_phase1 = [cp.sum(G_new_phase1) == cp.sum(G_original)]
constraints_phase1 = ramp_constraints_phase1 + energy_constraint_phase1 + non_negativity_constraint_phase1

# Solve the feasibility problem
problem_phase1 = cp.Problem(objective_phase1, constraints_phase1)
problem_phase1.solve()

# Extract the feasible solution for Phase 2
initial_solution_phase2 = G_new_phase1.value

# Variable for the new generation schedule, initialized with Phase 1's solution
G_new_phase2 = cp.Variable(len(data['Generation Schedule']), value=initial_solution_phase2)

# Objective for Phase 2 (refinement based on original schedule)
objective_terms_phase2 = cp.multiply(cp.abs(G_new_phase2 - G_original), G_original ** 2)
objective_phase2 = cp.Minimize(cp.sum(objective_terms_phase2))

# Constraints remain the same as Phase 1
ramp_diffs_phase2 = G_new_phase2[1:] - G_new_phase2[:-1]
ramp_constraints_phase2 = [cp.abs(ramp_diffs_phase2) <= 450]
non_negativity_constraint_phase2 = [G_new_phase2 >= 0]
energy_constraint_phase2 = [cp.sum(G_new_phase2) == cp.sum(G_original)]
constraints_phase2 = ramp_constraints_phase2 + energy_constraint_phase2 + non_negativity_constraint_phase2

# Solve the refinement problem
problem_phase2 = cp.Problem(objective_phase2, constraints_phase2)
problem_phase2.solve()

# Extract the refined solution
initial_solution_phase3 = G_new_phase2.value




# Visualization
plt.figure(figsize=(15, 7))
plt.plot(G_original, label='Original Schedule', color='blue')
plt.plot(initial_solution_phase2, label='Phase 1: Feasibility Solution', color='green', linestyle='--')
plt.plot(initial_solution_phase3, label='Phase 2: Refined Solution', color='red', linestyle='--')
# plt.plot(solution_phase3, label='Phase 3: Final Solution', color='black', linestyle='--')
plt.xlabel("Hour")
plt.ylabel("Generation Schedule")
plt.title("Original vs. Optimized Generation Schedules")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

k=2