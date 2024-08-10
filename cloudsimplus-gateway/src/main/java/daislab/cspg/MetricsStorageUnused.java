// package daislab.cspg;

// import org.apache.commons.collections4.queue.CircularFifoQueue;
// import org.apache.commons.lang3.ArrayUtils;

// import java.util.HashMap;
// import java.util.List;
// import java.util.Map;
// import java.util.Queue;
// import java.util.NoSuchElementException;

// public class MetricsStorageUnused {

    // private static final int hostsCount = settings.getDatacenterHostsCnt();
    // private static final int maxVmsCount = settings.getDatacenterHostsCnt() * settings.getHostPeCnt() / settings.getBasicVmPeCnt();
    // private static final int maxJobsCount = maxVmsCount * settings.getBasicVmPeCnt() / this.minJobPes;
    // private static final int minJobPes = 1;
    // private static final int datacenterMetricsCount = 6;
    // private static final int hostMetricsCount = 10;
    // private static final int vmMetricsCount = 7;
    // private static final int jobMetricsCount = 5;

    // private double[] datacenterMetrics;
    // private double[][] hostMetrics;
    // private double[][] vmMetrics;
    // private double[][] jobMetrics;

//     public MetricsStorageUnused() {
        // this.datacenterMetrics = new double[datacenterMetricsCount];
        // this.hostMetrics = new double[hostsCount][hostMetricsCount];
        // this.vmMetrics = new double[maxVmsCount][vmMetricsCount];
        // this.jobMetrics = new double[maxJobsCount][jobMetricsCount];
//     }

//     public void updateMetrics(
//         final double[] datacenterMetrics,
//         final double[][] hostMetrics,
//         final double[][] vmMetrics,
//         final double[][] jobMetrics
//     ) {
//         this.datacenterMetrics = datacenterMetrics;
//         this.hostMetrics = hostMetrics;
//         this.vmMetrics = vmMetrics;
//         this.jobMetrics = jobMetrics;
//     }

//     public void updateMetric(final String metricName, final Double value) {
//         if (!data.containsKey(metricName)) {
//             throw new NoSuchElementException("Unknown metric: " + metricName);
//         }
//         final CircularFifoQueue<Double> valuesQueue = data.get(metricName);
//         valuesQueue.add(value);
//     }

//     private void fillWithZeros(final Queue<Double> queue) {
//         for (int i = 0; i < historyLength; i++) {
//             queue.add(0.0);
//         }
//     }

//     public double[] metricValuesAsPrimitives(final String metricName) {
//         final CircularFifoQueue<Double> queue = data.get(metricName);
//         return ArrayUtils.toPrimitive(queue.toArray(new Double[0]));
//     }

//     public double getLastMetricValue(final String metricName) {
//         if (!data.containsKey(metricName)) {
//             throw new NoSuchElementException("Unknown metric: " + metricName);
//         }
//         final CircularFifoQueue<Double> valuesQueue = data.get(metricName);

//         return valuesQueue.get(valuesQueue.size() - 1);
//     }

//     public void clear() {
//         zeroAllMetrics();
//     }
// }
