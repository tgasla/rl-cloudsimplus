package daislab.cspg;

import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.List;
import java.util.Map;

public class SimulationFactory extends SimulationFactoryBase {

    private static final Type CLOUDLET_TYPE =
            new TypeToken<List<CloudletDescriptor>>() {}.getType();

    @Override
    protected Type getCloudletDescriptorsType() {
        return CLOUDLET_TYPE;
    }

    @Override
    protected CloudletDescriptor createSplitDescriptor(
            final CloudletDescriptor original, final int newId, final long mi, final int pes) {
        return new CloudletDescriptor(newId, original.getSubmissionDelay(), mi, pes);
    }

    @Override
    protected ISimulationSettings buildSettings(final Map<String, Object> params) {
        return new SimulationSettings(params);
    }

    @Override
    protected IWrappedSimulation buildSimulation(
            final String id, final ISimulationSettings settings,
            final List<CloudletDescriptor> jobs) {
        return new WrappedSimulation(id, settings, jobs);
    }
}
