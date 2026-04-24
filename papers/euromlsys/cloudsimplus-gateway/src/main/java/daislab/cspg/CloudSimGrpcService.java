package daislab.cspg;

import daislab.cspg.grpc.*;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * gRPC service implementation for euromlsys multi-DC simulation.
 */
public class CloudSimGrpcService extends CloudSimServiceGrpc.CloudSimServiceImplBase {

    private static final Logger LOGGER =
            LoggerFactory.getLogger(CloudSimGrpcService.class.getSimpleName());

    public CloudSimGrpcService() {
        LOGGER.info("CloudSimGrpcService constructor called");
    }

    private final Map<String, WrappedSimulation> simulations = new ConcurrentHashMap<>();
    private final SimulationFactory simulationFactory = new SimulationFactory();

    private volatile boolean shutdownRequested = false;

    @Override
    public void createSimulation(
            CreateRequest request,
            StreamObserver<CreateResponse> responseObserver) {
        try {
            LOGGER.info("gRPC createSimulation called");
            // euromlsys SimulationFactory.parseJson internally - pass raw JSON
            WrappedSimulation simulation =
                    simulationFactory.create(request.getParamsJson(), request.getJobsJson());
            String identifier = simulation.getIdentifier();
            simulations.put(identifier, simulation);

            CreateResponse response = CreateResponse.newBuilder()
                    .setSimId(identifier)
                    .build();
            responseObserver.onNext(response);
            responseObserver.onCompleted();
            LOGGER.info("Simulation {} created via gRPC", identifier);
        } catch (Exception e) {
            LOGGER.error("Error creating simulation", e);
            responseObserver.onError(
                    io.grpc.Status.INTERNAL
                            .withDescription(e.getMessage())
                            .asRuntimeException());
        }
    }

    @Override
    public void reset(ResetRequest request, StreamObserver<ResetResult> responseObserver) {
        String simId = request.getSimId();
        LOGGER.info("gRPC reset called for {}", simId);
        try {
            WrappedSimulation simulation = getValidSimulation(simId);
            SimulationResetResult javaResult = simulation.reset(request.getSeed());

            ResetResult grpcResult = ResetResult.newBuilder()
                    .setObservation(convertObservation(javaResult.getObservation()))
                    .setInfo(convertStepInfo(javaResult.getInfo()))
                    .build();

            responseObserver.onNext(grpcResult);
            responseObserver.onCompleted();
        } catch (Exception e) {
            LOGGER.error("Error resetting simulation {}", simId, e);
            responseObserver.onError(
                    io.grpc.Status.INTERNAL
                            .withDescription(e.getMessage())
                            .asRuntimeException());
        }
    }

    @Override
    public void step(StepRequest request, StreamObserver<StepResult> responseObserver) {
        String simId = request.getSimId();
        LOGGER.debug("Step request: simId={}, actionList.size={}", simId, request.getActionList().size());
        try {
            WrappedSimulation simulation = getValidSimulation(simId);
            if (request.getActionList().isEmpty()) {
                String errMsg = "Action list is empty for simId=" + simId;
                LOGGER.error(errMsg);
                responseObserver.onError(io.grpc.Status.INVALID_ARGUMENT
                        .withDescription(errMsg)
                        .asRuntimeException());
                return;
            }
            int[] actionArray = new int[request.getActionList().size()];
            for (int i = 0; i < actionArray.length; i++) {
                Object element = request.getActionList().get(i);
                if (element == null) {
                    String errMsg = "Action list has null element at index=" + i + " for simId=" + simId;
                    LOGGER.error(errMsg);
                    responseObserver.onError(io.grpc.Status.INVALID_ARGUMENT
                            .withDescription(errMsg)
                            .asRuntimeException());
                    return;
                }
                actionArray[i] = ((Number) element).intValue();
            }

            SimulationStepResult javaResult = simulation.step(actionArray);

            StepResult grpcResult = StepResult.newBuilder()
                    .setObservation(convertObservation(javaResult.getObservation()))
                    .setReward(javaResult.getReward())
                    .setTerminated(javaResult.isTerminated())
                    .setTruncated(javaResult.isTruncated())
                    .setInfo(convertStepInfo(javaResult.getInfo()))
                    .build();

            responseObserver.onNext(grpcResult);
            responseObserver.onCompleted();
        } catch (Exception e) {
            LOGGER.error("Error stepping simulation {}", simId, e);
            responseObserver.onError(
                    io.grpc.Status.INTERNAL
                            .withDescription(e.getMessage())
                            .asRuntimeException());
        }
    }

    @Override
    public void batchStep(BatchStepRequest request,
            StreamObserver<BatchStepResponse> responseObserver) {
        BatchStepResponse.Builder responseBuilder = BatchStepResponse.newBuilder();
        for (BatchStepRequest.StepItem item : request.getItemsList()) {
            String simId = item.getSimId();
            try {
                WrappedSimulation simulation = getValidSimulation(simId);
                int[] actionArray = new int[item.getActionList().size()];
                for (int i = 0; i < actionArray.length; i++) {
                    actionArray[i] = item.getActionList().get(i);
                }
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
                        .setTerminated(false)
                        .setTruncated(true)
                        .build());
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
            validateIdentifier(simId);
            WrappedSimulation simulation = simulations.remove(simId);
            if (simulation != null) {
                simulation.close();
            }

            if (simulations.isEmpty()) {
                LOGGER.info("Simulation {} closed. No simulations remaining. Signaling shutdown.", simId);
                shutdownRequested = true;
            } else {
                LOGGER.debug("Simulation {} closed. {} simulations still running.",
                        simId, simulations.size());
            }

            responseObserver.onNext(CloseResponse.newBuilder().build());
            responseObserver.onCompleted();
        } catch (Exception e) {
            LOGGER.error("Error closing simulation {}", simId, e);
            responseObserver.onError(
                    io.grpc.Status.INTERNAL
                            .withDescription(e.getMessage())
                            .asRuntimeException());
        }
    }

    @Override
    public void render(RenderRequest request, StreamObserver<RenderResponse> responseObserver) {
        try {
            WrappedSimulation simulation = getValidSimulation(request.getSimId());
            String renderData = simulation.render();
            responseObserver.onNext(
                    RenderResponse.newBuilder()
                            .setRenderDataJson(renderData)
                            .build());
            responseObserver.onCompleted();
        } catch (Exception e) {
            LOGGER.error("Error rendering simulation", e);
            responseObserver.onError(
                    io.grpc.Status.INTERNAL
                            .withDescription(e.getMessage())
                            .asRuntimeException());
        }
    }

    @Override
    public void ping(PingRequest request, StreamObserver<PongResponse> responseObserver) {
        responseObserver.onNext(PongResponse.newBuilder().setAlive(true).build());
        responseObserver.onCompleted();
    }

    boolean isShutdownRequested() {
        return shutdownRequested;
    }

    private WrappedSimulation getValidSimulation(String simId) {
        validateIdentifier(simId);
        return simulations.get(simId);
    }

    private void validateIdentifier(String simId) {
        if (!simulations.containsKey(simId)) {
            throw new IllegalArgumentException(
                    "Simulation with identifier: " + simId + " not found!");
        }
    }

    private static daislab.cspg.grpc.Observation convertObservation(Observation obs) {
        return daislab.cspg.grpc.Observation.newBuilder()
                .addAllInfrastructureObservation(intArrayToList(obs.getInfrastructureObservation()))
                .addAllJobsWaitingObservation(intArrayToList(obs.getJobsWaitingObservation()))
                .build();
    }

    private static daislab.cspg.grpc.StepInfo convertStepInfo(SimulationStepInfo info) {
        return daislab.cspg.grpc.StepInfo.newBuilder()
                .setJobsWaiting(info.getJobsWaiting())
                .setJobsPlaced(info.getJobsPlaced())
                .setJobsPlacedRatio(info.getJobsPlacedRatio())
                .setQualityRatio(info.getQualityRatio())
                .setDeadlineViolationRatio(info.getDeadlineViolationRatio())
                .addAllJobWaitTime(info.getJobWaitTime())
                .build();
    }

    private static List<Integer> intArrayToList(int[] arr) {
        java.util.List<Integer> list = new java.util.ArrayList<>(arr.length);
        for (int val : arr) list.add(val);
        return list;
    }
}
