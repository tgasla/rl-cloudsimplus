package daislab.cspg;

import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.apache.commons.lang3.ArrayUtils;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.NoSuchElementException;

public class MetricsStorage {

    private static final Double[] doubles = new Double[0];

    private final int historyLength;
    private final List<String> trackedMetrics;
    private Map<String, CircularFifoQueue<Double>> data = new HashMap<>();

    public MetricsStorage(final int historyLength, final List<String> trackedMetrics) {
        this.historyLength = historyLength;
        this.trackedMetrics = trackedMetrics;

        initializeStorage();
    }

    private void initializeStorage() {
        ensureMetricQueuesExist();
        zeroAllMetrics();
    }

    private void ensureMetricQueuesExist() {
        for(String metricName : trackedMetrics) {
            data.put(metricName, new CircularFifoQueue<>(historyLength));
        }
    }

    private void zeroAllMetrics() {
        for(String metricName : trackedMetrics) {
            final CircularFifoQueue<Double> metricQueue = data.get(metricName);
            metricQueue.clear();
            fillWithZeros(metricQueue);
        }
    }

    public void updateMetric(final String metricName, final Double value) {
        if (!data.containsKey(metricName)) {
            throw new NoSuchElementException("Unknown metric: " + metricName);
        }
        final CircularFifoQueue<Double> valuesQueue = data.get(metricName);
        valuesQueue.add(value);
    }

    private void fillWithZeros(final Queue<Double> queue) {
        for (int i = 0; i < historyLength; i++) {
            queue.add(0.0);
        }
    }

    public double[] metricValuesAsPrimitives(final String metricName) {
        final CircularFifoQueue<Double> queue = data.get(metricName);
        return ArrayUtils.toPrimitive(queue.toArray(doubles));
    }

    public double getLastMetricValue(final String metricName) {
        if (!data.containsKey(metricName)) {
            throw new NoSuchElementException("Unknown metric: " + metricName);
        }
        final CircularFifoQueue<Double> valuesQueue = data.get(metricName);

        return valuesQueue.get(valuesQueue.size() - 1);
    }

    public void clear() {
        zeroAllMetrics();
    }
}
