# Two-step optimization where the first steps try to optimize the problem with ramp constraints
# and total energy generation constraint included. The second step includes the peak constraints to the optimization
# problem. This particular script considers only morning and evening peaks, where  morning hours range is between
# 6:00 hrs to 10:00 hrs and evening peak range is between 16:00 hrs to 20:00 hrs.
import sys

import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../ramp_constraints.csv')

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
    sys.exit("Phase 1 optimization problem i infeasible with the given constraints")
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
    plt.show()

# Phase 2
# Define the time windows for morning and evening
morning_window = (6, 10)  # 6 AM to 10 AM
evening_window = (16, 20)  # 4 PM to 8 PM

# Determine the number of days in the data
num_days = data['Hour'].iloc[-1] // 24

# initialize lists to store the peak hours for each day
morning_peaks_indices_adjusted = []
evening_peaks_indices_adjusted = []

# Loop over each day to find the peaks within the morning and evening windows
for day in range(num_days + 1):  # Including potential partial day
    start_hour_morning = day * 24 + morning_window[0]
    end_hour_morning = day * 24 + morning_window[1]

    start_hour_evening = day * 24 + evening_window[0]
    end_hour_evening = day * 24 + evening_window[1]

    # Extract data for morning and evening windows for the current day
    morning_data_day = data[(data['Hour'] >= start_hour_morning) & (data['Hour'] < end_hour_morning)]
    evening_data_day = data[(data['Hour'] >= start_hour_evening) & (data['Hour'] < end_hour_evening)]

    # Check if there's data for the window, then find the peak hour
    if not morning_data_day.empty:
        morning_peak_index_day = morning_data_day['Generation Schedule'].idxmax()
        morning_peaks_indices_adjusted.append(morning_peak_index_day)
    if not evening_data_day.empty:
        evening_peak_index_day = evening_data_day['Generation Schedule'].idxmax()
        evening_peaks_indices_adjusted.append(evening_peak_index_day)

# actual peak constraints
morning_peak_constraints = [G_new[idx] == G_original[idx] for idx in morning_peaks_indices_adjusted]
evening_peak_constraints = [G_new[idx] == G_original[idx] for idx in evening_peaks_indices_adjusted]

upper_limit = 0.1
lower_limit = 0.001
epsilon = 0.001

while upper_limit - lower_limit > epsilon:
    tolerance = (upper_limit + lower_limit) / 2

    # For morning peaks
    morning_peak_constraints_relaxed_lower = [
        G_new[idx] >= (1 - tolerance) * G_original[idx]
        for idx in morning_peaks_indices_adjusted
    ]
    morning_peak_constraints_relaxed_upper = [
        G_new[idx] <= (1 + tolerance) * G_original[idx]
        for idx in morning_peaks_indices_adjusted
    ]

    # For evening peaks
    evening_peak_constraints_relaxed_lower = [
        G_new[idx] >= (1 - tolerance) * G_original[idx]
        for idx in evening_peaks_indices_adjusted
    ]
    evening_peak_constraints_relaxed_upper = [
        G_new[idx] <= (1 + tolerance) * G_original[idx]
        for idx in evening_peaks_indices_adjusted
    ]

    # Combine them into single lists for morning and evening respectively
    morning_peak_constraints_relaxed = morning_peak_constraints_relaxed_lower + morning_peak_constraints_relaxed_upper
    evening_peak_constraints_relaxed = evening_peak_constraints_relaxed_lower + evening_peak_constraints_relaxed_upper

    constraints_phase2 = ramp_constraints + energy_constraint + morning_peak_constraints_relaxed \
                         + evening_peak_constraints_relaxed

    problem_phase2 = cp.Problem(objective, constraints_phase2)
    problem_phase2.solve()

    if problem_phase2.status == 'optimal':
        upper_limit = tolerance
    else:
        lower_limit = tolerance

adjusted_generation_phase2 = G_new.value

if problem_phase2.status == "optimal":
    adjusted_generation_phase2 = G_new.value
    print("Phase 2 optimal solution found")
    plt.figure(figsize=(15, 6))
    plt.plot(data['Hour'], data['Generation Schedule'], label='Original Generation', color='blue')
    # plt.plot(data['Hour'], adjusted_generation_phase1, label='Adjusted Generation (Phase 1)', color='red',
    #          linestyle='--')
    plt.plot(data['Hour'], adjusted_generation_phase2, label='Adjusted Generation (Phase 2)', color='green',
             linestyle='--')
    plt.xlabel('Hour')
    plt.ylabel('Generation (MW)')
    plt.xticks(range(0, len(data["Hour"]) + 1, 24))  # setting x ticks at 24 hr intervals to indicate each day
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title('Original and Adjusted Generation Schedules (Phase 2)')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Phase 2 optimization problem is infeasible with the given constraints.")

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
plt.tight_layout()
plt.show()
k = 1
