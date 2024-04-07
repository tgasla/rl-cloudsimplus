package daislab.cspg;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.util.DataCloudTags;
import org.cloudsimplus.utilizationmodels.UtilizationModel;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;
import org.cloudsimplus.utilizationmodels.UtilizationModelPlanetLab;
import org.cloudsimplus.utilizationmodels.UtilizationModelStochastic;

import java.util.Objects;

public class CloudletDescriptor {
    private final int jobId;
    private final long submissionDelay;
    private final long mi;
    private final int numberOfCores;

    public CloudletDescriptor(
        final int jobId,
        final long submissionDelay,
        final long mi,
        final int numberOfCores
    ) {
        this.jobId = jobId;
        this.submissionDelay = submissionDelay;
        this.mi = mi;
        this.numberOfCores = numberOfCores;
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

    public int getNumberOfCores() {
        return numberOfCores;
    }

    @Override
    public boolean equals(final Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CloudletDescriptor that = (CloudletDescriptor) o;
        return getJobId() == that.getJobId()
            && getSubmissionDelay() == that.getSubmissionDelay()
            && getMi() == that.getMi()
            && getNumberOfCores() == that.getNumberOfCores();
    }

    @Override
    public int hashCode() {
        return Objects.hash(getJobId(), getSubmissionDelay(), getMi(), getNumberOfCores());
    }

    @Override
    public String toString() {
        return "CloudletDescriptor{"
            + "jobId=" + jobId
            + ", submissionDelay=" + submissionDelay
            + ", mi=" + mi
            + ", numberOfCores=" + numberOfCores
            + '}';
    }

    public Cloudlet toCloudlet() {
        Cloudlet cloudlet = new CloudletSimple(jobId, mi, numberOfCores)
            .setFileSize(DataCloudTags.DEFAULT_MTU)
            .setOutputSize(DataCloudTags.DEFAULT_MTU)
            .setUtilizationModelCpu(new UtilizationModelFull());
        cloudlet.setSubmissionDelay(submissionDelay);
        return cloudlet;
    }
}
