package daislab.cspg;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import daislab.cspg.grpc.*;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Type;
import java.util.List;
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
        System.err.println("UUUUU CloudSimGrpcService CONSTRUCTOR called UUUUU");
        LOGGER.info("UUUUU CloudSimGrpcService CONSTRUCTOR called UUUUU");
    }

    private final Map<String, WrappedSimulation> simulations = new ConcurrentHashMap<>();
    private final SimulationFactory simulationFactory = new SimulationFactory();
    private final Gson gson = new Gson();
    private final Type mapType = new TypeToken<Map<String, Object>>() {}.getType();

    private String runMode = "serial";
    private int experimentsFinishedCount = 0;
    private int numExperiments = 1;
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
            coerceNumericParams(params);
            WrappedSimulation simulation =
                    simulationFactory.create(params, request.getJobsJson());
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
        String actionStr = request.getActionList().toString();
        LOGGER.info("YYYYY step ENTRY simId={}, actionList.size={}, actionList={}", simId, actionSize, actionStr);
        System.err.println("YYYYY step ENTRY simId=" + simId + " actionSize=" + actionSize + " actionList=" + actionStr);
        try {
            WrappedSimulation simulation = getValidSimulation(simId);
            if (request.getActionList().isEmpty()) {
                String errMsg = "ZZZZZ ACTION_LIST_EMPTY_ZZZZZ simId=" + simId;
                System.err.println(errMsg);
                responseObserver.onError(io.grpc.Status.INVALID_ARGUMENT
                        .withDescription(errMsg)
                        .asRuntimeException());
                return;
            }
            int[] actionArray = new int[request.getActionList().size()];
            for (int i = 0; i < actionArray.length; i++) {
                Object element = request.getActionList().get(i);
                if (element == null) {
                    String errMsg = "ZZZZZ ACTION_LIST_HAS_NULL_ZZZZZ simId=" + simId + " index=" + i;
                    System.err.println(errMsg);
                    responseObserver.onError(io.grpc.Status.INVALID_ARGUMENT
                            .withDescription(errMsg)
                            .asRuntimeException());
                    return;
                }
                actionArray[i] = ((Number) element).intValue();
            }
            // Extra safety: if we got here with wrong size, fail explicitly
            if (actionArray.length < 4) {
                String errMsg = "ZZZZZ ACTION_ARRAY_TOO_SHORT_ZZZZZ simId=" + simId + " len=" + actionArray.length;
                System.err.println(errMsg);
                responseObserver.onError(io.grpc.Status.INVALID_ARGUMENT
                        .withDescription(errMsg)
                        .asRuntimeException());
                return;
            }
            LOGGER.info("YYYYY step PROCEEDING simId={}, actionArray.length={}", simId, actionArray.length);

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
            validateIdentifier(simId);
            WrappedSimulation simulation = simulations.remove(simId);
            if (simulation != null) {
                simulation.close();
            }

            experimentsFinishedCount++;
            LOGGER.debug("Simulation {} closed. {} simulations still running. "
                    + "Experiments finished: {}/{}",
                    simId, simulations.size(), experimentsFinishedCount, numExperiments);

            if (simulations.isEmpty() && experimentsFinishedCount >= numExperiments) {
                LOGGER.info("All experiments finished. Signaling shutdown.");
                shutdownRequested = true;
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

    /**
     * Defensively coerce any Number values in the params map to the type that
     * SimulationSettings expects (int or boolean). Gson parses all numbers as
     * Double by default when the map target is {@code Map<String, Object>},
     * causing ClassCastException on int casts.
     */
    private static void coerceNumericParams(Map<String, Object> params) {
        for (Map.Entry<String, Object> entry : params.entrySet()) {
            Object value = entry.getValue();
            if (value instanceof Number) {
                Number num = (Number) value;
                String key = entry.getKey();
                if (key.endsWith("_count") || key.endsWith("_pes") || key.endsWith("_length")
                        || key.equals("max_hosts") || key.equals("host_pes")
                        || key.equals("small_vm_pes") || key.equals("medium_vm_multiplier")
                        || key.equals("large_vm_multiplier") || key.equals("initial_s_vm_count")
                        || key.equals("initial_m_vm_count") || key.equals("initial_l_vm_count")
                        || key.equals("max_episode_length") || key.equals("max_job_pes")
                        || key.equals("host_pe_mips") || key.equals("host_ram")
                        || key.equals("host_storage") || key.equals("host_bw")
                        || key.equals("small_vm_ram") || key.equals("small_vm_storage")
                        || key.equals("small_vm_bw")) {
                    entry.setValue(num.intValue());
                }
            }
        }
    }
}
