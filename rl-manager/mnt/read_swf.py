"""Format as specified in
    http://www.cs.huji.ac.il/labs/parallel/workload/swf.html
"""
class SWFReader(object):
    def __init__(self):
        self.jobs = []

    def _as_cloudlet_descriptor_dict(self,
                                    job_id,
                                    submit_time,
                                    run_time,
                                    mips,
                                    allocated_cores):
        return {
            'jobId': job_id,
            'submissionDelay': submit_time,
            'mi': run_time * mips * allocated_cores,
            'numberOfCores': allocated_cores,
        }
    
    def read(self, filename, jobs_to_read=-1):
        with open(filename, 'r') as f:
            jobs_read = 0
            for line in f.readlines():
                if line.startswith(';'):
                    continue
                if jobs_to_read != -1 and jobs_read == jobs_to_read:
                    return self.jobs

                line_stripped = line.strip()
                line_splitted = line_stripped.split()

                job_id = int(line_splitted[0])
                submit_time = int(line_splitted[1])
                wait_time = int(line_splitted[2])
                run_time = int(line_splitted[3])
                allocated_cores = int(line_splitted[4])
                status = int(line_splitted[10])
                mips = 1250

                if (run_time > 0 and allocated_cores > 0 and status == 0):
                    cloudlet = self._as_cloudlet_descriptor_dict(
                        job_id,
                        submit_time,
                        run_time,
                        mips,
                        allocated_cores
                    )
                    self.jobs.append(cloudlet)
                    jobs_read += 1
        return self.jobs