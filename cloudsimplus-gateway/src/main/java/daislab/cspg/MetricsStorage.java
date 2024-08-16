package daislab.cspg;

public class MetricsStorage {
    private double[] datacenterMetrics;
    private double[][] hostMetrics;
    private double[][] vmMetrics;
    private double[][] jobMetrics;

    public MetricsStorage() {}

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
}
