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
    
    def read(self, filename, lines_to_read=-1):
        with open(filename, 'r') as f:
            line_num = 0
            for line in f.readlines():
                line_num += 1
                if line.startswith(';'):
                    continue
                if lines_to_read != -1 and line_num > lines_to_read:
                    return self.jobs

                stripped = line.strip()
                splitted = stripped.split()

                job_id = int(splitted[0])
                submit_time = int(splitted[1])
                wait_time = int(splitted[2])
                run_time = int(splitted[3])
                allocated_cores = int(splitted[4])
                status = int(splitted[10])

                mips = 1250

                if (int(run_time) > 0 and 
                        int(allocated_cores) > 0 and 
                        status == 0):
                    cloudlet = self._as_cloudlet_descriptor_dict(
                        job_id,
                        submit_time,
                        run_time,
                        mips,
                        allocated_cores
                    )
                    self.jobs.append(cloudlet)
        return self.jobs