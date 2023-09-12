from deap import base, creator, tools, algorithms
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('../ramp_constraints.csv')

# Define the time windows for morning and evening
morning_window = (6, 10)  # 6 AM to 10 AM
evening_window = (16, 20)  # 4 PM to 8 PM

# Determine the number of days in the data
num_days = data['Hour'].iloc[-1] // 24

# Re-initialize lists to store the peak hours for each day
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

all_peaks = morning_peaks_indices_adjusted + evening_peaks_indices_adjusted



# We want to minimize the objective function
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def initSchedule(ind_cls, original_schedule):
    return ind_cls([gen + random.uniform(-50, 50) for gen in original_schedule])


def evaluate(individual):
    deviation = sum(
        [(original - adjusted) ** 2 * original for original, adjusted in zip(data['Generation Schedule'], individual)])
    return deviation,


def feasible(individual):
    # Check ramp rate constraints
    for i in range(1, len(individual)):
        if abs(individual[i] - individual[i - 1]) > 450:
            return False

    # Check energy conservation
    if not np.isclose(sum(individual), sum(data['Generation Schedule'])):
        return False

    # Ensure non-negativity
    if any(gen < 0 for gen in individual):
        return False

    # Peak constraints
    for peak in all_peaks:
        if not np.isclose(individual[peak], data['Generation Schedule'].iloc[peak]):
            return False

    return True


toolbox = base.Toolbox()
toolbox.register("individual", initSchedule, creator.Individual, original_schedule=data['Generation Schedule'].values)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)  # Crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=50, indpb=0.2)  # Mutation
toolbox.register("select", tools.selTournament, tournsize=3)  # Selection
toolbox.register("evaluate", evaluate)
toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 7.0))  # Penalize infeasible solutions

population = toolbox.population(n=100)
ngen = 50  # Number of generations
mut_prob = 0.2  # Probability of mutation
cross_prob = 0.8  # Probability of crossover

# Run the genetic algorithm
algorithms.eaSimple(population, toolbox, cross_prob, mut_prob, ngen, stats=None, halloffame=None, verbose=True)

best_individual = tools.selBest(population, 1)[0]
print("Best Individual's Fitness:", best_individual.fitness.values)
print("Best Individual's Generation Schedule:", best_individual)

plt.figure(figsize=(12, 6))
plt.plot(data['Hour'], data['Generation Schedule'], label='Original Schedule', color='blue')
plt.plot(data['Hour'], best_individual, label='Best Schedule by GA', color='red', linestyle='--')
plt.xlabel("Hour")
plt.ylabel("Generation")
plt.title("Comparison of Original and Adjusted Generation Schedules")
plt.legend()
plt.grid(True)
plt.show()


k=1