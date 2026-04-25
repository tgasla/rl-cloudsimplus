import pandas as pd


def trace_to_csv(jobs: pd.DataFrame, filename="data.csv"):
    jobs.to_csv(filename)


def as_cloudlet_descriptor_dict(job_id, submit_time, mi, required_cores):
    """
    Base cloudlet descriptor with the common fields.
    Subclasses/papers can extend this with additional fields.
    """
    return {
        "jobId": job_id,
        "submissionDelay": submit_time,
        "mi": mi,
        "cores": required_cores,
    }


def csv_to_cloudlet_descriptor(filename):
    """
    Load jobs from a CSV with columns: job_id, arrival_time, mi, required_cores, location, ...
    Returns list of cloudlet descriptors with base schema.
    """
    jobs = []
    df = pd.read_csv(filename, sep=",")
    for i in df.index:
        cloudlet = as_cloudlet_descriptor_dict(
            int(df.job_id[i]),
            int(df.arrival_time[i]),
            int(df.mi[i]),
            int(df.required_cores[i]),
        )
        # Handle optional location field for euromlsys-style traces
        if "location" in df.columns:
            cloudlet["location"] = df.location[i]
        # Handle optional delay_sensitivity field
        if "delay_sensitivity" in df.columns:
            cloudlet["delaySensitivity"] = df.delay_sensitivity[i]
        jobs.append(cloudlet)
    return jobs


def swf_to_cloudlet_descriptor(filename, jobs_to_read=None, relative_submission_delay=True):
    """
    Format as specified in http://www.cs.huji.ac.il/labs/parallel/workload/swf.html
    Returns base cloudlet descriptors with job_id, submit_time, mi, required_cores.
    """
    jobs = []
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
            required_cores = int(line_splitted[4])
            status = int(line_splitted[10])
            mi = run_time * mips * required_cores

            if status != 0 or run_time <= 0 or required_cores <= 0:
                continue

            if relative_submission_delay:
                if previous_submit_time is None:
                    previous_submit_time = submit_time
                    submit_time = 0
                else:
                    original_submit_time = submit_time
                    submit_time -= previous_submit_time
                    previous_submit_time = original_submit_time

            cloudlet = as_cloudlet_descriptor_dict(job_id, submit_time, mi, required_cores)
            jobs.append(cloudlet)
            jobs_read += 1
    return jobs