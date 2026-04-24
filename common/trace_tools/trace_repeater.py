import pandas as pd

df = pd.read_csv("three_60max_8maxcores.csv")

# Repeat the job trace 5 times, adjusting job_id and arrival_time
num_repeats = 5
offset = 31  # Arrival time offset for each repetition

all_repeats = []
for i in range(num_repeats):
    repetition = df.copy()
    repetition["job_id"] += (i + 1) * len(df)
    repetition["arrival_time"] += (i + 1) * offset
    all_repeats.append(repetition)

# Combine the original and repeated job traces
final_df = pd.concat([df] + all_repeats, ignore_index=True)
final_df = final_df.sort_values(by="arrival_time", ascending=True)

# Reset the index (optional, but often helpful after sorting)
final_df.reset_index(drop=True, inplace=True)
final_df.to_csv(f"three_60max_8maxcores_{num_repeats}repeated.csv", index=False)
