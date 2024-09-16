import matplotlib.pyplot as plt
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, help="The trace data to print")
    args = parser.parse_args()
    df = pd.read_csv(args.trace)
    plt.figure(figsize=(10, 5))
    plt.bar(df.arrival_time, df.allocated_cores)
    plt.xlabel("Job Arrival Time")
    plt.ylabel("Jobs Arrived")
    plt.savefig(args.trace.split(".")[0] + ".png")


if __name__ == "__main__":
    main()
