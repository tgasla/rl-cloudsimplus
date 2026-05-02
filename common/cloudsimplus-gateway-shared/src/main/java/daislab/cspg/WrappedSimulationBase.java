package daislab.cspg;

import java.util.List;
import java.util.stream.Collectors;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Unified abstract base for RL-driven CloudSim simulations.
 *
 * Defines the shared RL orchestration loop (reset/step/close) and declares
 * abstract methods for domain-specific behavior. Concrete domains extend this
 * class and implement the abstract methods.
 *
 * The three strategy interfaces (IStateExtractor, IActionDecoder, IRewardCalculator)
 * are injected at construction and handle domain-specific state/action/reward logic.
 */
public abstract class WrappedSimulationBase<OBS, STEP_INFO, STEP_RESULT, RESET_RESULT> {

    protected final Logger LOGGER = LoggerFactory.getLogger(getClass());

    protected final String identifier;
    protected final ISimulationSettings settings;
    protected final List<CloudletDescriptor> initialJobsDescriptors;

    protected ICloudSimProxy cloudSimProxy;
    protected int currentStep;

    protected IStateExtractor stateExtractor;
    protected IActionDecoder actionDecoder;
    protected IRewardCalculator rewardCalculator;

    protected WrappedSimulationBase(
            final String identifier,
            final ISimulationSettings settings,
            final List<CloudletDescriptor> jobs,
            final IStateExtractor stateExtractor,
            final IActionDecoder actionDecoder,
            final IRewardCalculator rewardCalculator) {
        this.identifier = identifier;
        this.settings = settings;
        this.initialJobsDescriptors = jobs;
        this.stateExtractor = stateExtractor;
        this.actionDecoder = actionDecoder;
        this.rewardCalculator = rewardCalculator;
        LOGGER.info("Creating simulation: {}", identifier);
    }

    // ============== Shared RL loop ==============

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

    public RESET_RESULT reset(final long seed) {
        // seed is ignored (used only for reproducibility in distributed training)
        LOGGER.info("Reset initiated");
        LOGGER.info("job count: " + initialJobsDescriptors.size());

        currentStep = 0;

        List<Cloudlet> cloudlets = initialJobsDescriptors.stream()
                .map(CloudletDescriptor::toCloudlet).collect(Collectors.toList());
        cloudSimProxy = createCloudSimProxy(cloudlets);

        int[] infraObs = stateExtractor.extractState();
        int[] secondaryObs = extractSecondaryObservation();
        OBS observation = buildObservation(infraObs, secondaryObs);

        STEP_INFO info = buildResetStepInfo();

        return buildResetResult(observation, info);
    }

    public STEP_RESULT step(final int[] action) {
        validateSimulationReset();
        currentStep++;
        LOGGER.info("Step {} starting", currentStep);

        final int[] actionResult = actionDecoder.decodeAction(action);

        cloudSimProxy.runOneTimestep();

        boolean terminated = !cloudSimProxy.isRunning();
        boolean truncated = !terminated && (currentStep >= settings.getMaxEpisodeLength());

        double reward = rewardCalculator.calculateReward();

        LOGGER.info("Step {} finished", currentStep);
        LOGGER.debug("Terminated: {}, Truncated: {}", terminated, truncated);
        LOGGER.debug("Length of future events queue: {}", cloudSimProxy.getNumberOfFutureEvents());
        if (terminated || truncated) {
            LOGGER.info("Simulation ended. Jobs finished: {}/{}",
                    cloudSimProxy.getFinishedJobsCount(),
                    initialJobsDescriptors.size());
        }

        int[] infraObs = stateExtractor.extractState();
        int[] secondaryObs = extractSecondaryObservation();
        OBS observation = buildObservation(infraObs, secondaryObs);

        STEP_INFO info = buildStepInfo(actionResult, terminated, truncated);

        return buildStepResult(observation, reward, terminated, truncated, info);
    }

    public String render() {
        return "";
    }

    public int getCurrentStep() {
        return currentStep;
    }

    public String getIdentifier() {
        return identifier;
    }

    public double clock() {
        return cloudSimProxy.clock();
    }

    // ============== Abstract methods for domain-specific behavior ==============

    /** Create domain-specific simulation proxy with initial cloudlets */
    protected abstract ICloudSimProxy createCloudSimProxy(List<Cloudlet> cloudlets);

    /** Extract secondary observation (e.g., jobs waiting, cores waiting) */
    protected abstract int[] extractSecondaryObservation();

    /** Build domain-specific observation from infrastructure and secondary observations */
    protected abstract OBS buildObservation(int[] infraObs, int[] secondaryObs);

    /** Build SimulationStepInfo for reset */
    protected abstract STEP_INFO buildResetStepInfo();

    /** Build SimulationStepInfo for step result */
    protected abstract STEP_INFO buildStepInfo(int[] actionResult,
            boolean terminated, boolean truncated);

    /** Build SimulationResetResult from observation and step info */
    protected abstract RESET_RESULT buildResetResult(OBS observation, STEP_INFO info);

    /** Build SimulationStepResult from observation, reward, terminated, truncated, and step info */
    protected abstract STEP_RESULT buildStepResult(OBS observation, double reward,
            boolean terminated, boolean truncated, STEP_INFO info);
}
