# Commands to Produce the Plots

## Trace File Plot

### Files Used

#### python script: [trace-plotter-stacked.py](trace_tools/trace-plotter-stacked.py)
#### synthetic trace data file: [three_peaks_max30_multicores_long_noise_8maxcores](rl-manager/mnt/traces/three_peaks_max30_multicores_long_noise_8maxcores.csv)

### Command Used

```
python3 trace_tools/trace-plotter-stacked.py --trace rl-manager/mnt/traces/three_peaks_max30_multicores_long_noise_8maxcores.csv --filetype pdf
```

The synthetic trace data file was created by running [trace_generator.py](trace_tools/trace_generator.py) and [add_noise_to_trace.py](trace_tools/add_noise_to_trace.py), but both scripts are non-deterministic and I did not keep the seed for their randomness.

## Scatter Plot

First run the experiment for all heuristics to gather the data

I will provide the config.yml for all experiments

Then, run this to gather data (take median values only for job wait reward, running vm cores reward, unutilized vm cores reward)

```
python3 logs/individual_reward_seed_aggregator.py --log_files ../usenix_experiments/minimize-allocated/16hosts/best_episode_<num>.csv --multiplier <3_for_heuristics|4_for_drl>
```

Then, when you have all data, run `${HOME}/Desktop/mobihoc25/2dall-2legend.py` (TODO: this is not in this repo, it is saved locally) with filling the values of medians to the corresponding places.

## Job Waiting Times Boxplot

Use the exact same directories, but instead of `best_episode_<num>.csv`, now use the `job_wait_time.csv` file and the following script

```
python3 logs/job_wait_in_sec_compare_algos_boxplots.py ../usenix_experiments/minimize-queue/16hosts/job_wait_time.csv ../usenix_experiments/daistwo_maskable_train/job_wait_time.csv ../usenix_experiments/minimize-unutilized/16hosts/job_wait_time.csv ../usenix_experiments/minimize-allocated/16hosts/job_wait_time.csv
```

But, the DRL agent data is slightly changed, probably I aggregated all 5 seeds, so I need to find the script which aggregates those seeds first (and also which 5 seeds were aggregated). Maybe I just get all job wait times from 5 seeds and put them in google sheets and then took the 5 values required for the boxplot and put them.


## Barplot 300k from scratch vs 300k transferred
I am not sure, but I maybe used the script `tensorboard_max_rew_single_file` and got the max reward for a seed. I repeated it for all 5 seeds and took the average (in google sheets). Then, used this numbers and feed it into `${HOME}/Desktop/mobihoc25/transfer_no_empty_bars.py`.


## Barplot 500k timesteps tranferred vs 2M from scratched (asssuming approx. 95% converged)

Repeat the same process as above, but the file that produces the plots this time is `${HOME}/Desktop/mobihoc25/reward_of_total.py`.
