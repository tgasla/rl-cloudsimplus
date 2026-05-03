package daislab.cspg;

import org.cloudsimplus.brokers.DatacenterBroker;

/**
 * Minimal interface for RL simulation control.
 * Both domain's CloudSimProxy implementations satisfy this interface.
 */
public interface ICloudSimProxy {
    boolean isRunning();
    void terminate();
    void runOneTimestep();
    long getNumberOfFutureEvents();
    double clock();
    DatacenterBroker getBroker();
    long getFinishedJobsCount();
}
