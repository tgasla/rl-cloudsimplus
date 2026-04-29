package daislab.cspg;

import java.util.Arrays;
import lombok.Data;

@Data
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

    public void clear() {
        Arrays.fill(this.datacenterMetrics, 0);
        Arrays.stream(this.hostMetrics).forEach(array -> Arrays.fill(array, 0));
        Arrays.stream(this.vmMetrics).forEach(array -> Arrays.fill(array, 0));
        Arrays.stream(this.jobMetrics).forEach(array -> Arrays.fill(array, 0));
    }
}
