package daislab.cspg;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import daislab.cspg.grpc.*;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Type;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * gRPC service implementation that wraps MultiSimulationEnvironment's logic.
 * Each call is isolated - simulations are keyed by sim_id, allowing one
 * JVM to serve multiple concurrent simulation workers.
 */
public class CloudSimGrpcService extends CloudSimServiceGrpc.CloudSimServiceImplBase {

    private static final Logger LOGGER =
            LoggerFactory.getLogger(CloudSimGrpcService.class.getSimpleName());

    public CloudSimGrpcService() {
        LOGGER.info("CloudSimGrpcService constructor called");
    }

    private final ConcurrentHashMap<String, WrappedSimulation> simulations = new ConcurrentHashMap<>();
    private final SimulationFactory simulationFactory = new SimulationFactory();
    private final Gson gson = new Gson();
    private final Type mapType = new TypeToken<Map<String, Object>>() {}.getType();
    private volatile boolean shutdownRequested = false;

    @Override
    public void createSimulation(
            CreateRequest request,
            StreamObserver<CreateResponse> responseObserver) {
        try {
            LOGGER.info("gRPC createSimulation called");
            // Parse the JSON params into a Map (what SimulationFactory expects)
            Map<String, Object> params = gson.fromJson(request.getParamsJson(), mapType);
            // Defensively coerce any Number values to their target types to avoid
            // ClassCastException when Gson returns Double instead of Integer
            GrpcServiceHelper.coerceNumericParams(params);
            // Convert Map back to JSON string for SimulationFactory (which parses JSON internally)
            String paramsJson = gson.toJson(params);
            WrappedSimulation simulation =
                    simulationFactory.create(paramsJson, request.getJobsJson());
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

            // Convert Java result to gRPC message
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
        int actionSize = request.getActionList().size();
        LOGGER.debug("Step request: simId={}, actionList.size={}", simId, actionSize);
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
            if (actionArray.length < 4) {
                String errMsg = "Action array too short: length=" + actionArray.length + " for simId=" + simId;
                LOGGER.error(errMsg);
                responseObserver.onError(io.grpc.Status.INVALID_ARGUMENT
                        .withDescription(errMsg)
                        .asRuntimeException());
                return;
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
                // Return a "truncated" result to indicate failure for this item
                // without failing the entire batch
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
            getValidSimulation(simId); // validates simId exists
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

    // ─────────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────────

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
        daislab.cspg.grpc.Observation.Builder builder =
                daislab.cspg.grpc.Observation.newBuilder()
                        .addAllInfrastructureObservation(
                                intArrayToList(obs.getInfrastructureObservation()))
                        .setJobCoresWaitingObservation(obs.getJobCoresWaitingObservation());
        return builder.build();
    }

    private static daislab.cspg.grpc.StepInfo convertStepInfo(SimulationStepInfo info) {
        daislab.cspg.grpc.StepInfo.Builder builder =
                daislab.cspg.grpc.StepInfo.newBuilder()
                        .setJobWaitReward(info.getJobWaitReward())
                        .setRunningVmCoresReward(info.getRunningVmCoresReward())
                        .setUnutilizedVmCoresReward(info.getUnutilizedVmCoresReward())
                        .setInvalidReward(info.getInvalidReward())
                        .setIsValid(info.isValid())
                        .addAllJobWaitTime(info.getJobWaitTime())  // List<Double>
                        .setUnutilizedVmCoreRatio(info.getUnutilizedVmCoreRatio())
                        .addAllObservationTreeArray(
                                convertIntArray(info.getObservationTreeArrayAsList()))
                        .setHostAffected(info.getHostAffected())
                        .setCoresChanged(info.getCoresChanged());
        return builder.build();
    }

    private static java.util.List<Integer> intArrayToList(int[] arr) {
        java.util.List<Integer> list = new java.util.ArrayList<>(arr.length);
        for (int val : arr) list.add(val);
        return list;
    }

    private static java.util.List<Integer> convertIntArray(java.util.List<Integer> list) {
        return list;
    }
}