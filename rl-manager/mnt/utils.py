import numpy as np
import pandas as pd
from typing import Dict, List, Union, Literal

class FilenameFormatter(object):
    @staticmethod
    def _millify(num):
        num = float(f"{num:.3f}")
        magnitude = 0
        suffix = ['', 'K', 'M', 'B', 'T']
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000
        return f"{num:.0f}" + suffix[magnitude]
    
    @staticmethod
    def create_filename_id(
        algorithm_str,
        reward_job_wait_coef,
        reward_vm_cost_coef,
        reward_invalid_coef,
        pretrain_env, 
        pretrain_timesteps,
        transfer_env=None,
        transfer_timesteps=None
    ):
        filename_id = (
            f"{algorithm_str}"
            f"_Q{reward_job_wait_coef}"
            f"_U{reward_vm_cost_coef}"
            f"_I{reward_invalid_coef}"
            f"_p{pretrain_env}"
            f"_{FilenameFormatter._millify(pretrain_timesteps)}"
        )
        if transfer_env is not None:
            assert transfer_timesteps is not None

            filename_id = (
                f"{filename_id}"
                f"_r{transfer_env}"
                f"_{FilenameFormatter._millify(transfer_timesteps)}"
            )
        return filename_id
    

class WorkloadUtils(object):
    # @staticmethod
    # def _dl_to_ld(dl):
    #     '''
    #     Turns a dictionary of lists to a list of dictionaries
    #     '''
    #     return [dict(zip(dl,t)) for t in zip(*dl.values())]
        
    # @staticmethod
    # def _ld_to_dl(ld):
    #     '''
    #     Turns a list of dictionaries to a dictionary of lists
    #     '''
    #     common_keys = set.intersection(*map(set,ld))
    #     return {k: [dic[k] for dic in ld] for k in common_keys}

    # def _ld_to_csv(jobs, filename="data.csv"):
    #     with open(filename, "w") as file: 
    #         w = csv.DictWriter(file, jobs[0].keys(), encoding="utf8", newline="")
    #         w.writeheader()
    #         w.writerows(jobs)

    @staticmethod
    def to_csv(
        jobs: Union[List[Dict], Dict, pd.DataFrame],
        filename="data.csv"
    ):
        if isinstance(jobs, pd.DataFrame):
            jobs.to_csv(filename)
        # elif isinstance(jobs, List[Dict]):
        #     WorkloadUtils._ld_to_csv(jobs, filename)

    @staticmethod
    def read_csv(filename="data.csv"):
        jobs = []
        df = pd.read_csv(filename, sep=",")
        for i in df.index:
            cloudlet = CloudletDescriptor.as_cloudlet_descriptor_dict(
                int(df.job_id[i]),
                int(df.arrival_time[i]),
                int(df.mi[i]),
                int(df.allocated_cores[i])
            )
            jobs.append(cloudlet)
        return jobs

    @staticmethod
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
        # type: Literal[
        #     "dataframe", 
        #     "list_of_dict", 
        #     "dict_of_list"
        # ] = "dataframe"
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
            WorkloadUtils.to_csv(job_trace, csv_filename)
        return job_trace
    

class SWFUtils(object):
    """
        Format as specified in
        http://www.cs.huji.ac.il/labs/parallel/workload/swf.html
    """
    @staticmethod
    def read_swf(filename, jobs_to_read=None, relative_submission_delay=True):
        jobs = []
        # mips = 1250
        # controls how long jobs run
        mips = 10
        previous_submit_time = None
        with open(filename, "r") as f:
            jobs_read = 0
            for line in f.readlines():
                if line.startswith(";"):
                    continue
                if jobs_to_read is not None and jobs_read == jobs_to_read:
                    return jobs

                line_stripped = line.strip()
                line_splitted = line_stripped.split()

                job_id = int(line_splitted[0])
                submit_time = int(line_splitted[1])
                run_time = int(line_splitted[3])
                allocated_cores = int(line_splitted[4])
                status = int(line_splitted[10])
                mi = run_time * mips * allocated_cores

                if status != 0 or run_time <= 0 or allocated_cores <= 0:
                    continue

                if relative_submission_delay:
                    if previous_submit_time is None:
                        # The first job's relative submission time is 0
                        previous_submit_time = submit_time
                        submit_time = 0
                    else:
                        # Update submit_time to be relative to the previous job
                        original_submit_time = submit_time
                        submit_time -= previous_submit_time
                        previous_submit_time = original_submit_time
                    
                cloudlet = CloudletDescriptor.as_cloudlet_descriptor_dict(
                    job_id,
                    submit_time,
                    mi,
                    allocated_cores
                )
                jobs.append(cloudlet)
                jobs_read += 1
        return jobs
    
class CloudletDescriptor(object):
    @staticmethod
    def as_cloudlet_descriptor_dict(
        job_id,
        submit_time,
        mi,
        allocated_cores
    ):
        return {
            "jobId": job_id,
            "submissionDelay": submit_time,
            "mi": mi,
            "numberOfCores": allocated_cores,
        }