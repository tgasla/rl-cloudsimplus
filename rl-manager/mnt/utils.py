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

 
class SWFReader(object):
    """
        Format as specified in
        http://www.cs.huji.ac.il/labs/parallel/workload/swf.html
    """
    @staticmethod
    def _as_cloudlet_descriptor_dict(
        job_id,
        submit_time,
        run_time,
        mips,
        allocated_cores
    ):
        return {
            'jobId': job_id,
            'submissionDelay': submit_time,
            'mi': run_time * mips * allocated_cores,
            'numberOfCores': allocated_cores,
        }
    
    @staticmethod
    def swf_read(filename, jobs_to_read=None, relative_submission_delay=True):
        jobs = []
        previous_submit_time = None
        with open(filename, 'r') as f:
            jobs_read = 0
            for line in f.readlines():
                if line.startswith(';'):
                    continue
                if jobs_to_read is not None and \
                    jobs_read == jobs_to_read:
                    return jobs

                line_stripped = line.strip()
                line_splitted = line_stripped.split()

                job_id = int(line_splitted[0])
                submit_time = int(line_splitted[1])
                run_time = int(line_splitted[3])
                allocated_cores = int(line_splitted[4])
                status = int(line_splitted[10])
                # mips = 1250
                # controls how long jobs run
                mips = 10000

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
                    
                cloudlet = SWFReader._as_cloudlet_descriptor_dict(
                    job_id,
                    submit_time,
                    run_time,
                    mips,
                    allocated_cores
                )
                jobs.append(cloudlet)
                jobs_read += 1
        return jobs