import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../ramp_constraints.csv')

# Analyzing the data to identify morning and evening peaks

# First, let's take a look at the average generation for each hour across all days to get an idea of the general
# daily pattern
average_generation_by_hour = data.groupby(data['Hour'] % 24)['Generation Schedule'].mean()

# Plotting the average generation by hour to visually inspect morning and evening peaks
plt.figure(figsize=(10, 5))
average_generation_by_hour.plot()
plt.title('Average Generation by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Generation (MW)')
plt.grid(True)
plt.tight_layout()
plt.show()

# capturing morning and evening peaks
# Define the time windows for morning and evening
morning_window = (6, 10)  # 6 AM to 10 AM
evening_window = (16, 20)  # 4 PM to 8 PM

# Determine the number of days in the data
num_days = data['Hour'].iloc[-1] // 24

# Re-initialize lists to store the peak hours for each day
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
plt.title('Original Generation Schedule with Identified Morning and Evening Peaks')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

k=1