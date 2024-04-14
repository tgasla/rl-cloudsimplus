import pandas as pd

# Simulating reading from a CSV file, you can replace this with pd.read_csv('filename.csv')

data = "1000jobs_1lambda_1000mi.csv"

# Using StringIO to simulate reading from a file
df = pd.read_csv(data)

# Initialize an empty DataFrame to hold the expanded rows
expanded_df = pd.DataFrame()

# Initialize job_id counter
job_id_counter = 0

# Process each row in the original DataFrame
for index, row in df.iterrows():
    # Number of times to duplicate the row
    num_duplicates = row['allocated_cores']
    
    # Create new rows with adjusted values
    new_rows = pd.DataFrame({
        'job_id': range(job_id_counter, job_id_counter + num_duplicates),
        'arrival_time': row['arrival_time'],
        'mi': [row['mi'] / num_duplicates] * num_duplicates,
        'allocated_cores': [1] * num_duplicates
    })
    
    # Update job_id_counter for the next set of rows
    job_id_counter += num_duplicates
    
    # Append new rows to the expanded DataFrame
    expanded_df = expanded_df.append(new_rows)

# Reset index of the new DataFrame to make it clean
expanded_df.reset_index(drop=True, inplace=True)

# Display or save the expanded DataFrame
print(expanded_df)
# You can save to a new CSV file using expanded_df.to_csv('output_filename.csv', index=False)
