import pandas as pd
import numpy as np

df = pd.read_csv("three_60max_8maxcores.csv")

# Analyze the maximum number of jobs arriving per second
max_jobs_per_second = df["arrival_time"].value_counts().max()

# Create a new DataFrame to hold the modified trace
new_trace = []

# Silent period (no jobs)
silent_periods = [
    5,
    6,
    7,
    8,
    9,
    16,
    17,
    18,
    19,
]  # Example silent periods, you can customize
silent_period_set = set(silent_periods)

# Iterate over 0 to 30 seconds to create the new trace
job_id = 0
for t in range(31):  # From 0 to 30 seconds
    if t in silent_period_set:
        continue  # Skip silent periods

    # Burst periods (intense job arrivals)
    if t % 10 == 0:  # Example: Every 10th second has a burst
        num_jobs = max_jobs_per_second * 4
    else:  # Normal job arrivals
        num_jobs = max_jobs_per_second

    # Generate jobs for this second
    for _ in range(num_jobs):
        mi = np.random.choice([50000, 100000, 150000, 200000, 250000])
        required_cores = np.random.choice([1, 2, 4, 8])
        new_trace.append([job_id, t, mi, required_cores])
        job_id += 1

# Create a DataFrame for the new trace
new_df = pd.DataFrame(
    new_trace, columns=["job_id", "arrival_time", "mi", "required_cores"]
)

# Save the modified trace to a CSV file (optional)
new_df.to_csv("three_30max_8maxcores.csv", index=False)
