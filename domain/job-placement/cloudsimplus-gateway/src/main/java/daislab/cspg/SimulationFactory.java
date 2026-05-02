package daislab.cspg;

import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;

public class SimulationFactory extends SimulationFactoryBase {

    private static final Type CLOUDLET_TYPE =
            new TypeToken<List<CloudletDescriptorWithLocation>>() {}.getType();

    @Override
    protected Type getCloudletDescriptorsType() {
        return CLOUDLET_TYPE;
    }

    @Override
    protected CloudletDescriptor createSplitDescriptor(
            final CloudletDescriptor original, final int newId, final long mi, final int pes) {
        CloudletDescriptorWithLocation cdl = (CloudletDescriptorWithLocation) original;
        return new CloudletDescriptorWithLocation(newId, original.getSubmissionDelay(), mi, pes,
                cdl.getLocation(), cdl.getDelaySensitivity(), cdl.getDeadline());
    }

    public synchronized WrappedSimulation create(
            final String paramsAsJson, final String jobsAsJson) {
        String identifier = "Sim" + simulationsRunning++;
        ISimulationSettings settings = new SimulationSettings(parseParamsJson(paramsAsJson));
        LOGGER.info("Simulation settings dump:\n{}", settings);
        List<CloudletDescriptor> jobs = loadJobsFromJson(jobsAsJson);
        if (settings.isSplitLargeJobs()) {
            LOGGER.info("Splitting large jobs");
            jobs = splitLargeJobs(jobs, settings.getMaxJobPes());
        }
        return new WrappedSimulation(identifier, settings, new ArrayList<>(jobs));
    }
}
