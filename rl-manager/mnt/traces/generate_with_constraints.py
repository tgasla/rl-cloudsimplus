import random
import csv

# Initialize variables
total_jobs = 100
arrival_time = 1
current_job_count = 0
jobs = []
core_options = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
mips = 60  # MIPS for all cores
max_jobs_per_arrival = 4  # Maximum number of jobs per arrival time
job_id = 0


# Function to calculate MI based on cores
def calculate_mi(cores):
    # To make jobs run between 1-5 seconds, calculate MI for a given number of cores
    time_per_job = (500 / cores) / mips  # Time taken for the job to finish
    desired_time = random.uniform(1, 5)  # Desired time between 1 and 5 seconds
    mi = int(500 * desired_time * mips / (500 / cores))  # Adjust MI based on cores
    return mi, time_per_job


# Function to generate a job
def generate_job(job_id, arrival_time):
    cores = random.choice(core_options)  # Choose a random number of cores
    mi, time_taken = calculate_mi(cores)  # Calculate MI and time taken
    delay_sensitivity = random.choice(["moderate", "critical", "tolerant"])
    deadline = random.randint(0, 5)
    location = random.choice(["micro_dc_ucd", "micro_dc_dcu", "micro_dc_aau"])

    job = [job_id, arrival_time, mi, cores, location, delay_sensitivity, deadline]
    return job


# Generate jobs
while current_job_count < total_jobs:
    # Random number of jobs for this arrival time
    jobs_at_this_time = random.randint(
        1, min(max_jobs_per_arrival, total_jobs - current_job_count)
    )

    for _ in range(jobs_at_this_time):
        job = generate_job(job_id, arrival_time)
        jobs.append(job)
        job_id += 1
        current_job_count += 1

        if current_job_count >= total_jobs:
            break

    arrival_time += 1  # Increment arrival time

# Save to CSV
header = [
    "job_id",
    "arrival_time",
    "mi",
    "required_cores",
    "location",
    "delay_sensitivity",
    "deadline",
]

with open("euromlsys_jobs_three_locs.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(jobs)

print("Dataset generated and saved as 'euromlsys_jobs_three_locs.csv'.")
