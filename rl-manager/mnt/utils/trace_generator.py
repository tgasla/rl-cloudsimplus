import numpy as np
import pandas as pd
from .trace_utils import trace_to_csv

def generate_job_trace(
    n_jobs=100,
    distribution="poisson",
    lambda_poisson=1,
    mu_gaussian=1,
    sigma_gaussian=1,
    mi_multiplier=1_000,
    max_cores=8,
    seed=0,
    noise=0,
    csv_filename=None
):
    np.random.seed(seed)
    # Generate arrival times based on the chosen distribution
    if distribution =="poisson":
        arrival_times = np.cumsum(np.random.poisson(lambda_poisson, n_jobs))
    elif distribution == "gaussian":
        arrival_times = np.abs(np.cumsum(np.random.normal(mu_gaussian, sigma_gaussian, n_jobs)))
    else:
        raise ValueError("Unsupported distribution. Choose either 'poisson' or 'gaussian'.")

    # MI values between 0 and 1
    mis = np.round(mi_multiplier * np.random.uniform(0, 1, n_jobs)).astype(int)

    # Generating allocated_cores to be divisible by 2 and not exceed max_cores
    allocated_cores = np.random.choice([i for i in range(2, max_cores + 1, 2)], n_jobs)

    if noise:
        # Adding Gaussian noise to the arrival times
        # Mean = 0, Std Dev = lambda/2
        noise = np.random.normal(0, lambda_poisson / 2, n_jobs)
        arrival_times = np.max(arrival_times + noise, 0)

    # dictionaries of list
    job_trace = {
        "job_id": list(range(n_jobs)),
        "arrival_time": arrival_times,
        "mi": mis,
        "allocated_cores": allocated_cores
    }

    # if type == "list_of_dict":
    #     job_trace = WorkloadUtils._dl_to_ld(job_trace)
    # elif type == "dataframe":
    job_trace = pd.DataFrame.from_dict(job_trace, orient='index').transpose()
    if csv_filename:
        trace_to_csv(job_trace, csv_filename)
    return job_trace