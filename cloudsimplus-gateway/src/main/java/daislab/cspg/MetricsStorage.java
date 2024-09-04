package daislab.cspg;

import java.util.Arrays;

public class MetricsStorage {
    private double[] datacenterMetrics;
    private double[][] hostMetrics;
    private double[][] vmMetrics;
    private double[][] jobMetrics;

    public MetricsStorage(final int datacenterMetricsCount, final int hostMetricsCount,
            final int vmMetricsCount, final int jobMetricsCount, final int hostsCount,
            final int maxVmsCount, final int maxJobsCount) {
        this.datacenterMetrics = new double[datacenterMetricsCount];
        this.hostMetrics = new double[hostsCount][hostMetricsCount];
        this.vmMetrics = new double[maxVmsCount][vmMetricsCount];
        this.jobMetrics = new double[maxJobsCount][jobMetricsCount];
    }

    public void setDatacenterMetrics(double[] datacenterMetrics) {
        this.datacenterMetrics = datacenterMetrics;
    }

    public void setHostMetrics(double[][] hostMetrics) {
        this.hostMetrics = hostMetrics;
    }

    public void setVmMetrics(double[][] vmMetrics) {
        this.vmMetrics = vmMetrics;
    }

    public void setJobMetrics(double[][] jobMetrics) {
        this.jobMetrics = jobMetrics;
    }

    public double[] getDatacenterMetrics() {
        return this.datacenterMetrics;
    }

    public double[][] getHostMetrics() {
        return this.hostMetrics;
    }

    public double[][] getVmMetrics() {
        return this.vmMetrics;
    }

    public double[][] getJobMetrics() {
        return this.jobMetrics;
    }

    public void clear() {
        Arrays.fill(this.datacenterMetrics, 0);
        Arrays.stream(this.hostMetrics).forEach(array -> Arrays.fill(array, 0));
        Arrays.stream(this.vmMetrics).forEach(array -> Arrays.fill(array, 0));
        Arrays.stream(this.jobMetrics).forEach(array -> Arrays.fill(array, 0));
    }
}
