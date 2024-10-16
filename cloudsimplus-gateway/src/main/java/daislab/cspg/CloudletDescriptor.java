package daislab.cspg;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.util.DataCloudTags;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;

import java.util.Objects;

public class CloudletDescriptor {
    private final int jobId;
    private final long submissionDelay;
    private final long mi;
    private final int cores;

    public CloudletDescriptor(final int jobId, final long submissionDelay, final long mi,
            final int cores) {
        this.jobId = jobId;
        this.submissionDelay = submissionDelay;
        this.mi = mi;
        this.cores = cores;
    }

    public int getJobId() {
        return jobId;
    }

    public long getSubmissionDelay() {
        return submissionDelay;
    }

    public long getMi() {
        return mi;
    }

    public int getCores() {
        return cores;
    }

    @Override
    public boolean equals(final Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        CloudletDescriptor job = (CloudletDescriptor) obj;
        return getJobId() == job.getJobId() && getSubmissionDelay() == job.getSubmissionDelay()
                && getMi() == job.getMi() && getCores() == job.getCores();
    }

    @Override
    public int hashCode() {
        return Objects.hash(getJobId(), getSubmissionDelay(), getMi(), getCores());
    }

    @Override
    public String toString() {
        return "CloudletDescriptor{" + "jobId=" + jobId + ", submissionDelay=" + submissionDelay
                + ", mi=" + mi + ", cores=" + cores + '}';
    }

    public Cloudlet toCloudlet() {
        Cloudlet cloudlet = new CloudletSimple(jobId, mi, cores)
                .setFileSize(DataCloudTags.DEFAULT_MTU).setOutputSize(DataCloudTags.DEFAULT_MTU)
                .setUtilizationModelCpu(new UtilizationModelFull());
        cloudlet.setSubmissionDelay(submissionDelay);
        return cloudlet;
    }
}
