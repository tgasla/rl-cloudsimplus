package daislab.cspg;

import java.util.List;
import java.util.stream.Collectors;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.vms.Vm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Unified abstract base for RL-driven CloudSim simulations.
 *
 * Provides the shared {@link #reset(long)} loop and infrastructure plumbing
 * (settings, identifier, simulation proxy, step counter). Each domain owns
 * its own {@link #step(int[])} implementation because action handling, reward
 * computation, and step-info construction differ enough across domains that
 * a one-size-fits-all template only added indirection.
 *
 * Common RL-level helpers (unutilised VM core ratios) live here so domains
 * can use them without duplicating the logic.
 */
public abstract class WrappedSimulationBase implements IWrappedSimulation {

    protected final Logger LOGGER = LoggerFactory.getLogger(getClass());

    protected final String identifier;
    protected final ISimulationSettings settings;
    protected final List<CloudletDescriptor> initialJobsDescriptors;

    protected ICloudSimProxy cloudSimProxy;
    protected int currentStep;

    protected WrappedSimulationBase(
            final String identifier,
            final ISimulationSettings settings,
            final List<CloudletDescriptor> jobs) {
        this.identifier = identifier;
        this.settings = settings;
        this.initialJobsDescriptors = jobs;
        LOGGER.info("Creating simulation: {}", identifier);
    }

    // ============== Shared lifecycle ==============

    @Override
    public void close() {
        LOGGER.info("Terminating simulation...");
        if (cloudSimProxy.isRunning()) {
            cloudSimProxy.terminate();
        }
    }

    public void validateSimulationReset() {
        if (cloudSimProxy == null) {
            throw new IllegalStateException(
                    "Simulation not reset! Please call the reset() function before calling step!");
        }
    }

    @Override
    public SimulationResetResult reset(final long seed) {
        // seed is ignored (used only for reproducibility in distributed training)
        LOGGER.info("Reset initiated");
        LOGGER.info("job count: {}", initialJobsDescriptors.size());

        currentStep = 0;

        List<Cloudlet> cloudlets = initialJobsDescriptors.stream()
                .map(CloudletDescriptor::toCloudlet).collect(Collectors.toList());
        cloudSimProxy = createCloudSimProxy(cloudlets);

        Observation observation = buildObservation(extractInfrastructureObservation(),
                extractSecondaryObservation());
        SimulationStepInfo info = buildResetStepInfo();

        return buildResetResult(observation, info);
    }

    @Override
    public String render() {
        return "";
    }

    public int getCurrentStep() {
        return currentStep;
    }

    @Override
    public String getIdentifier() {
        return identifier;
    }

    public double clock() {
        return cloudSimProxy.clock();
    }

    // ============== Concrete shared helpers ==============

    protected Observation buildObservation(int[] infraObs, int[] secondaryObs) {
        return new Observation(infraObs, secondaryObs);
    }

    protected SimulationStepInfo buildResetStepInfo() {
        return new SimulationStepInfo();
    }

    protected SimulationResetResult buildResetResult(Observation observation,
            SimulationStepInfo info) {
        return new SimulationResetResult(observation, info);
    }

    /** Ratio of free PEs across all running VMs to total running VM PEs. */
    protected double getUnutilizedVmCoreRatio() {
        List<Vm> vmList = cloudSimProxy.getBroker().getVmExecList();
        long unutilized = getUnutilizedVmCores(vmList);
        long running = getRunningVmCores(vmList);
        return running > 0 ? ((double) unutilized / running) : 0.0;
    }

    protected long getUnutilizedVmCores(List<Vm> vmList) {
        return vmList.parallelStream().map(Vm::getExpectedFreePesNumber).reduce(0L, Long::sum);
    }

    protected long getRunningVmCores(List<Vm> vmList) {
        return vmList.parallelStream().map(Vm::getPesNumber).reduce(0L, Long::sum);
    }

    // ============== Domain-specific abstract methods ==============

    /** Create domain-specific simulation proxy with initial cloudlets. */
    protected abstract ICloudSimProxy createCloudSimProxy(List<Cloudlet> cloudlets);

    /** Domain-specific step loop: decode action, advance clock, compute reward, build result. */
    @Override
    public abstract SimulationStepResult step(int[] action);

    /** Extract infrastructure observation (state) from the simulation. */
    protected abstract int[] extractInfrastructureObservation();

    /** Extract secondary observation (e.g., jobs waiting, cores waiting). */
    protected abstract int[] extractSecondaryObservation();
}
