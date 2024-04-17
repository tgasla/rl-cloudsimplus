package daislab.cspg;

public class TimeMeasurement {
    public static long startTiming() {
        return System.nanoTime();
    }

    public static long calculateElapsedTime(long startTime) {
        long endTime = System.nanoTime();
        return endTime - startTime;
    }
}
