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