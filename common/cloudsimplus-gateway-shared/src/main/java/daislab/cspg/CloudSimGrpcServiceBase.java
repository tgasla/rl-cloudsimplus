package daislab.cspg;

import daislab.cspg.grpc.*;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Shared gRPC service implementation for all CloudSim RL domains.
 * Handles all RPC wiring; domain subclasses provide only the SimulationFactory.
 *
 * Unified conversion works because Observation and SimulationStepInfo already carry
 * all domain fields; each domain populates its slice and leaves the rest at defaults.
 */
public abstract class CloudSimGrpcServiceBase extends CloudSimServiceGrpc.CloudSimServiceImplBase {

    private static final Logger LOGGER =
            LoggerFactory.getLogger(CloudSimGrpcServiceBase.class.getSimpleName());

    private final GrpcServiceDelegate delegate = new GrpcServiceDelegate();
    private final SimulationFactoryBase factory;

    protected CloudSimGrpcServiceBase(SimulationFactoryBase factory) {
        this.factory = factory;
        LOGGER.info("CloudSimGrpcService constructor called");
    }

    /** Hook for domain-specific action array validation. Default: no-op. */
    protected void validateActionArray(int[] action, String simId) {}

    @Override
    public void createSimulation(
            CreateRequest request,
            StreamObserver<CreateResponse> responseObserver) {
        try {
            LOGGER.info("gRPC createSimulation called");
            IWrappedSimulation simulation = factory.create(request.getParamsJson(), request.getJobsJson());
            String identifier = simulation.getIdentifier();
            delegate.putSimulation(identifier, simulation);
            responseObserver.onNext(CreateResponse.newBuilder().setSimId(identifier).build());
            responseObserver.onCompleted();
            LOGGER.info("Simulation {} created via gRPC", identifier);
        } catch (Exception e) {
            LOGGER.error("Error creating simulation", e);
            responseObserver.onError(io.grpc.Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void reset(ResetRequest request, StreamObserver<ResetResult> responseObserver) {
        String simId = request.getSimId();
        LOGGER.info("gRPC reset called for {}", simId);
        try {
            IWrappedSimulation simulation = delegate.getValidSimulation(simId);
            SimulationResetResult javaResult = simulation.reset(request.getSeed());
            responseObserver.onNext(ResetResult.newBuilder()
                    .setObservation(convertObservation(javaResult.getObservation()))
                    .setInfo(convertStepInfo(javaResult.getInfo()))
                    .build());
            responseObserver.onCompleted();
        } catch (Exception e) {
            LOGGER.error("Error resetting simulation {}", simId, e);
            responseObserver.onError(io.grpc.Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void step(StepRequest request, StreamObserver<StepResult> responseObserver) {
        String simId = request.getSimId();
        LOGGER.debug("Step request: simId={}, actionList.size={}", simId, request.getActionList().size());
        try {
            IWrappedSimulation simulation = delegate.getValidSimulation(simId);
            int[] actionArray;
            try {
                actionArray = delegate.parseActionArray(request.getActionList(), simId);
            } catch (IllegalArgumentException e) {
                LOGGER.error(e.getMessage());
                responseObserver.onError(io.grpc.Status.INVALID_ARGUMENT
                        .withDescription(e.getMessage()).asRuntimeException());
                return;
            }
            try {
                validateActionArray(actionArray, simId);
            } catch (IllegalArgumentException e) {
                LOGGER.error(e.getMessage());
                responseObserver.onError(io.grpc.Status.INVALID_ARGUMENT
                        .withDescription(e.getMessage()).asRuntimeException());
                return;
            }
            SimulationStepResult javaResult = simulation.step(actionArray);
            responseObserver.onNext(StepResult.newBuilder()
                    .setObservation(convertObservation(javaResult.getObservation()))
                    .setReward(javaResult.getReward())
                    .setTerminated(javaResult.isTerminated())
                    .setTruncated(javaResult.isTruncated())
                    .setInfo(convertStepInfo(javaResult.getInfo()))
                    .build());
            responseObserver.onCompleted();
        } catch (Exception e) {
            LOGGER.error("Error stepping simulation {}", simId, e);
            responseObserver.onError(io.grpc.Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void batchStep(BatchStepRequest request,
            StreamObserver<BatchStepResponse> responseObserver) {
        BatchStepResponse.Builder responseBuilder = BatchStepResponse.newBuilder();
        for (BatchStepRequest.StepItem item : request.getItemsList()) {
            String simId = item.getSimId();
            try {
                IWrappedSimulation simulation = delegate.getValidSimulation(simId);
                int[] actionArray = delegate.parseActionArray(item.getActionList(), simId);
                SimulationStepResult javaResult = simulation.step(actionArray);
                responseBuilder.addResults(StepResult.newBuilder()
                        .setObservation(convertObservation(javaResult.getObservation()))
                        .setReward(javaResult.getReward())
                        .setTerminated(javaResult.isTerminated())
                        .setTruncated(javaResult.isTruncated())
                        .setInfo(convertStepInfo(javaResult.getInfo()))
                        .build());
            } catch (Exception e) {
                LOGGER.error("batchStep item failed for simId={}: {}", simId, e.getMessage());
                responseBuilder.addResults(StepResult.newBuilder()
                        .setTerminated(false).setTruncated(true).build());
            }
        }
        responseObserver.onNext(responseBuilder.build());
        responseObserver.onCompleted();
    }

    @Override
    public void close(CloseRequest request, StreamObserver<CloseResponse> responseObserver) {
        String simId = request.getSimId();
        LOGGER.info("gRPC close called for {}", simId);
        try {
            delegate.validateIdentifier(simId);
            IWrappedSimulation simulation = delegate.removeSimulation(simId);
            if (simulation != null) {
                simulation.close();
            }
            if (delegate.isSimulationsEmpty()) {
                LOGGER.info("Simulation {} closed. No simulations remaining. Signaling shutdown.", simId);
                delegate.requestShutdown();
            } else {
                LOGGER.debug("Simulation {} closed. {} simulations still running.",
                        simId, delegate.simulationCount());
            }
            responseObserver.onNext(CloseResponse.newBuilder().build());
            responseObserver.onCompleted();
        } catch (Exception e) {
            LOGGER.error("Error closing simulation {}", simId, e);
            responseObserver.onError(io.grpc.Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void render(RenderRequest request, StreamObserver<RenderResponse> responseObserver) {
        try {
            IWrappedSimulation simulation = delegate.getValidSimulation(request.getSimId());
            responseObserver.onNext(RenderResponse.newBuilder()
                    .setRenderDataJson(simulation.render()).build());
            responseObserver.onCompleted();
        } catch (Exception e) {
            LOGGER.error("Error rendering simulation", e);
            responseObserver.onError(io.grpc.Status.INTERNAL
                    .withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void ping(PingRequest request, StreamObserver<PongResponse> responseObserver) {
        responseObserver.onNext(PongResponse.newBuilder().setAlive(true).build());
        responseObserver.onCompleted();
    }

    boolean isShutdownRequested() {
        return delegate.isShutdownRequested();
    }

    private static daislab.cspg.grpc.Observation convertObservation(Observation obs) {
        return daislab.cspg.grpc.Observation.newBuilder()
                .addAllInfrastructureObservation(
                        GrpcServiceDelegate.intArrayToList(obs.getInfrastructureObservation()))
                .setJobCoresWaitingObservation(obs.getJobCoresWaitingObservation())
                .addAllJobsWaitingObservation(
                        GrpcServiceDelegate.intArrayToList(obs.getJobsWaitingObservation()))
                .build();
    }

    private static daislab.cspg.grpc.StepInfo convertStepInfo(SimulationStepInfo info) {
        return daislab.cspg.grpc.StepInfo.newBuilder()
                .setJobWaitReward(info.getJobWaitReward())
                .setRunningVmCoresReward(info.getRunningVmCoresReward())
                .setUnutilizedVmCoresReward(info.getUnutilizedVmCoresReward())
                .setInvalidReward(info.getInvalidReward())
                .setIsValid(info.isValid())
                .addAllJobWaitTime(info.getJobWaitTime())
                .setUnutilizedVmCoreRatio(info.getUnutilizedVmCoreRatio())
                .addAllObservationTreeArray(info.getObservationTreeArrayAsList())
                .setHostAffected(info.getHostAffected())
                .setCoresChanged(info.getCoresChanged())
                .setJobsWaiting(info.getJobsWaiting())
                .setJobsPlaced(info.getJobsPlaced())
                .setJobsPlacedRatio(info.getJobsPlacedRatio())
                .setQualityRatio(info.getQualityRatio())
                .setDeadlineViolationRatio(info.getDeadlineViolationRatio())
                .build();
    }
}
