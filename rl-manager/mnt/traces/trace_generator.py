import csv
import random
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--max-entropy", type=int)
    parser.add_argument("--last-slot", type=int)
    parser.add_argument("--filename", type=str)
    args = parser.parse_args()
    # Open CSV file for writing
    with open(args.filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["job_id", "arrival_time", "mi", "allocated_cores"])

        arrival_time = 0
        mi = 10  # MI is constant for now
        job_id = 0

        for _ in range(args.last_slot):
            # Generate allocated_cores based on uniform distribution [-max_entropy, max_entropy]
            allocated_cores = random.uniform(-args.max_entropy, args.max_entropy)
            # Convert negative allocated_cores to 0 (no jobs arrived at this slot)
            allocated_cores = max(0, round(allocated_cores))

            # Write the row to CSV if at least 1 core is allocated (i.e., job arrived)
            arrival_time += 1
            if allocated_cores > 0:
                job_id += 1
                writer.writerow([job_id, arrival_time, mi, allocated_cores])


if __name__ == "__main__":
    main()
