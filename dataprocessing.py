# To process and plot relevant data for visual inspection
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

data = pd.read_csv('ramp_constraints.csv')

# plot original generation schedule
file_path = "plots\\hourly_generation_schedule.png"
plt.figure(figsize=(12, 6))
plt.plot(data["Hour"], data["Generation Schedule"], linestyle='-', color='blue')
plt.title("Hourly Generation Schedule")
plt.xlabel("Hour")
plt.ylabel("Generation Schedule")
plt.xticks(range(0, len(data["Hour"]) + 1, 24))  # setting x ticks at 24 hr intervals to indicate each day
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(file_path)


# First, let's take a look at the average generation for each hour across all days to get an idea of the general
# daily pattern
average_generation_by_hour = data.groupby(data['Hour'] % 24)['Generation Schedule'].mean()

# Plotting the average generation by hour to visually inspect morning and evening peaks
file_path = "plots\\average_generation_schedule.png"
plt.figure(figsize=(10, 5))
average_generation_by_hour.plot()
plt.title('Average Generation by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Generation (MW)')
plt.xticks(range(0, len(data["Hour"]) + 1, 24))  # setting x ticks at 24 hr intervals to indicate each day
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(file_path)

# capturing morning and evening peaks
# Define the time windows for morning and evening
file_path = "plots\\am_pm_peaks_generation_schedule.png"
morning_window = (6, 10)  # 6 AM to 10 AM
evening_window = (16, 20)  # 4 PM to 8 PM

# Determine the number of days in the data
num_days = data['Hour'].iloc[-1] // 24

# initialize lists to store the peak hours for each day
morning_peaks_indices = []
evening_peaks_indices = []

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
        morning_peaks_indices.append(morning_peak_index_day)
    if not evening_data_day.empty:
        evening_peak_index_day = evening_data_day['Generation Schedule'].idxmax()
        evening_peaks_indices.append(evening_peak_index_day)

# Plotting the original data with the adjusted identified morning and evening peaks
plt.figure(figsize=(15, 6))
plt.plot(data['Hour'], data['Generation Schedule'], label='Generation Schedule', color='blue')
plt.scatter(data['Hour'].iloc[morning_peaks_indices],
            data['Generation Schedule'].iloc[morning_peaks_indices], color='red', label='Morning Peaks', s=50)
plt.scatter(data['Hour'].iloc[evening_peaks_indices],
            data['Generation Schedule'].iloc[evening_peaks_indices], color='green', label='Evening Peaks',
            s=50)
plt.xlabel('Hour')
plt.ylabel('Generation (MW)')
plt.xticks(range(0, len(data["Hour"]) + 1, 24))  # setting x ticks at 24 hr intervals to indicate each day
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.title('Original Generation Schedule with Identified Morning and Evening Peaks')
plt.legend()
plt.tight_layout()
plt.savefig(file_path)

# plot all peaks
file_path = "plots\\all_local_peaks_generation_schedule.png"
peaks_indices, _ = find_peaks(data['Generation Schedule'])

peak_indices, _ = find_peaks(data['Generation Schedule'].values)
plt.figure(figsize=(15, 6))
plt.plot(data['Hour'], data['Generation Schedule'], label='Generation Schedule', color='blue')
plt.scatter(data['Hour'].iloc[peak_indices], data['Generation Schedule'].iloc[peak_indices], color='red',
            label='Detected Peaks', s=50)
plt.xlabel('Hour')
plt.ylabel('Generation (MW)')
plt.xticks(range(0, len(data["Hour"]) + 1, 24))  # setting x ticks at 24 hr intervals to indicate each day
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.title('Original Generation Schedule with Detected Peaks')
plt.legend()
plt.tight_layout()
plt.savefig(file_path)

k=1