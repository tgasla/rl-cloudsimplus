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
        pretrain_env, 
        pretrain_timesteps,
        transfer_env=None,
        transfer_timesteps=None
    ):
        filename_id = (
            f"{algorithm_str}"
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
    def swf_read(filename, jobs_to_read=None):
        jobs = []
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
                # wait_time = int(line_splitted[2])
                run_time = int(line_splitted[3])
                allocated_cores = int(line_splitted[4])
                status = int(line_splitted[10])
                mips = 1250

                if (run_time > 0 and allocated_cores > 0 and status == 0):
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