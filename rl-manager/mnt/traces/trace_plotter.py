import argparse
import pandas as pd
import matplotlib.pyplot as plt


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--trace", type=str, help="The trace data to print")
#     args = parser.parse_args()
#     df = pd.read_csv(args.trace)
#     plt.figure(figsize=(10, 5))
#     plt.bar(df.arrival_time, df.allocated_cores)
#     plt.xlabel("Job Arrival Time")
#     plt.ylabel("Job Cores")
#     plt.savefig(args.trace.split(".")[0] + ".png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, help="The trace data to print")
    parser.add_argument(
        "--count-cores",
        type=bool,
        default=True,
        help="Count the number of cores instead of jobs",
    )
    args = parser.parse_args()

    # Read the trace data
    df = pd.read_csv(args.trace)

    # Group by arrival time and count the number of jobs
    if not args.count_cores:
        grouped = df.groupby("arrival_time").size()
    else:
        grouped = df.groupby("arrival_time")["allocated_cores"].sum()

    # Plot the bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(grouped.index, grouped.values, color="skyblue", edgecolor="black")
    plt.xlabel("Job Arrival Time")
    plt.title("Job Arrival Distribution")
    if not args.count_cores:
        plt.ylabel("Number of Jobs")
    else:
        plt.ylabel("Number of Cores")

    # Save the plot
    output_filename = args.trace.split(".")[0] + ".png"
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")


if __name__ == "__main__":
    main()
