import numpy as np
import pandas as pd
import argparse
from utils.trace_utils import trace_to_csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distribution",
        type=str,
        default="poisson"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0
    )
    parser.add_argument(
        "--id-start-from",
        type=int,
        default=0
    )
    parser.add_argument(
        "--lambda-poisson",
        type=int,
        default=1
    )
    parser.add_argument(
        "--mu-gaussian",
        type=int,
        default=1
    )
    parser.add_argument(
        "--sigma-gaussian",
        type=int,
        default=1
    )
    parser.add_argument(
        "--jobs",
        type=int
    )
    parser.add_argument(
        "--mi-multiplier",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--max-cores",
        type=int,
        default=8
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )
    parser.add_argument(
        "--noise",
        type=bool,
        default=False
    )
    args = parser.parse_args()
    generate_job_trace(args)


def generate_job_trace(args):
    np.random.seed(args.seed)
    # Generate arrival times based on the chosen distribution
    if args.distribution =="poisson":
        arrival_times = args.start_from + np.cumsum(np.random.poisson(args.lambda_poisson, args.jobs))
    elif args.distribution == "gaussian":
        arrival_times = args.start_from + np.abs(np.cumsum(np.random.normal(args.mu_gaussian, args.sigma_gaussian, args.jobs)))
    else:
        raise ValueError("Unsupported distribution. Choose either 'poisson' or 'gaussian'.")

    # MI values between 0 and 1
    mis = np.round(args.mi_multiplier * np.random.uniform(0, 1, args.jobs)).astype(int)

    core_range = [1] + list(range(2, args.max_cores + 1, 2))

    # Generating allocated_cores to be divisible by 2 and not exceed max_cores
    allocated_cores = np.random.choice([i for i in core_range], args.jobs)

    if args.noise:
        # Adding Gaussian noise to the arrival times
        # Mean = 0, Std Dev = lambda/2
        args.noise = np.random.normal(0, args.lambda_poisson / 2, args.jobs)
        arrival_times = np.max(arrival_times + args.noise, 0)

    # dictionaries of list
    job_trace = {
        "job_id": list(range(args.id_start_from, args.id_start_from + args.jobs)),
        "arrival_time": arrival_times,
        "mi": mis,
        "allocated_cores": allocated_cores
    }

    # if type == "list_of_dict":
    #     job_trace = WorkloadUtils._dl_to_ld(job_trace)
    # elif type == "dataframe":
    csv_filename = f"{args.jobs}jobs_{args.lambda_poisson}lambda_{args.mi_multiplier}mi_{args.start_from}start.csv"
    job_trace = pd.DataFrame.from_dict(job_trace, orient='index').transpose()
    job_trace = job_trace.set_index("job_id")
    if csv_filename:
        trace_to_csv(job_trace, csv_filename)
    return job_trace


if __name__ == "__main__":
    main()