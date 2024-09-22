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
    private final int coresNumber;

    public CloudletDescriptor(final int jobId, final long submissionDelay, final long mi,
            final int coresNumber) {
        this.jobId = jobId;
        this.submissionDelay = submissionDelay;
        this.mi = mi;
        this.coresNumber = coresNumber;
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

    public int getCoresNumber() {
        return coresNumber;
    }

    @Override
    public boolean equals(final Object obj) {
        if (this == obj)
            return true;
        if (obj == null || getClass() != obj.getClass())
            return false;
        CloudletDescriptor job = (CloudletDescriptor) obj;
        return getJobId() == job.getJobId() && getSubmissionDelay() == job.getSubmissionDelay()
                && getMi() == job.getMi() && getCoresNumber() == job.getCoresNumber();
    }

    @Override
    public int hashCode() {
        return Objects.hash(getJobId(), getSubmissionDelay(), getMi(), getCoresNumber());
    }

    @Override
    public String toString() {
        return "CloudletDescriptor{" + "jobId=" + jobId + ", submissionDelay=" + submissionDelay
                + ", mi=" + mi + ", numberOfCores=" + coresNumber + '}';
    }

    public Cloudlet toCloudlet() {
        Cloudlet cloudlet = new CloudletSimple(jobId, mi, coresNumber)
                .setFileSize(DataCloudTags.DEFAULT_MTU).setOutputSize(DataCloudTags.DEFAULT_MTU)
                .setUtilizationModelCpu(new UtilizationModelFull());
        cloudlet.setSubmissionDelay(submissionDelay);
        return cloudlet;
    }
}
