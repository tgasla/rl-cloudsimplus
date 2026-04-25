"""CloudSim gRPC Client — unified for vm_management and job_placement RL problems."""
import grpc


# ─────────────────────────────────────────────────────────────────────────────
# RL Problem detection (mirrors SimulationSettingsBuilder.detectRlProblem)
# ─────────────────────────────────────────────────────────────────────────────
def _detect_rl_problem(params: dict) -> str:
    """Detect RL problem type from params dict.

    Detection order:
      1. Explicit `rl_problem` key -> use directly
      2. `host_count` / `hosts_count` present -> VM_MANAGEMENT
      3. `datacenters` / `max_datacenters` present -> JOB_PLACEMENT
      4. Default: VM_MANAGEMENT (backward compatibility)
    """
    explicit = params.get("rl_problem")
    if explicit:
        s = str(explicit).lower()
        if s in ("vm_management", "job_placement"):
            return s

    if params.get("host_count") or params.get("hosts_count"):
        return "vm_management"
    if params.get("datacenters") is not None or params.get("max_datacenters") is not None:
        return "job_placement"

    return "vm_management"


# ─────────────────────────────────────────────────────────────────────────────
# Unified proto import (serves both RL problem types)
# ─────────────────────────────────────────────────────────────────────────────
from .protos.unified import cloudsimplus_pb2 as pb2
from .protos.unified import cloudsimplus_pb2_grpc as pb2_grpc


class CloudSimGrpcClient:
    """Wrapper around CloudSimServiceStub for CloudSim Plus gRPC communication.

    Supports both VM_MANAGEMENT (main paper) and JOB_PLACEMENT (euromlsys paper)
    by auto-detecting the RL problem from params and using the unified proto
    (which contains fields for both problem types).
    """

    def __init__(self, host="localhost", port=50051):
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = pb2_grpc.CloudSimServiceStub(self.channel)

    def create_simulation(self, params_json: str, jobs_json: str, rl_problem: str = None) -> str:
        """Create a new simulation, returns sim_id.

        rl_problem is detected from params_json if not provided explicitly.
        """
        request = pb2.CreateRequest(params_json=params_json, jobs_json=jobs_json)
        response = self.stub.createSimulation(request)
        return response.sim_id

    def _step_info_to_dict(self, info, rl_problem: str):
        """Convert StepInfo protobuf to dict based on RL problem type."""
        if not info:
            return {}
        if rl_problem == "vm_management":
            return {
                "job_wait_reward": info.job_wait_reward if hasattr(info, 'job_wait_reward') else 0.0,
                "running_vm_cores_reward": info.running_vm_cores_reward if hasattr(info, 'running_vm_cores_reward') else 0.0,
                "unutilized_vm_cores_reward": info.unutilized_vm_cores_reward if hasattr(info, 'unutilized_vm_cores_reward') else 0.0,
                "invalid_reward": info.invalid_reward if hasattr(info, 'invalid_reward') else 0.0,
                "is_valid": info.is_valid if hasattr(info, 'is_valid') else True,
                "job_wait_time": list(info.job_wait_time) if hasattr(info, 'job_wait_time') else [],
                "unutilized_vm_core_ratio": info.unutilized_vm_core_ratio if hasattr(info, 'unutilized_vm_core_ratio') else 0.0,
                "observation_tree_array": list(info.observation_tree_array) if hasattr(info, 'observation_tree_array') else [],
                "host_affected": info.host_affected if hasattr(info, 'host_affected') else 0,
                "cores_changed": info.cores_changed if hasattr(info, 'cores_changed') else 0,
            }
        # job_placement
        return {
            "jobs_waiting": info.jobs_waiting if hasattr(info, 'jobs_waiting') else 0,
            "jobs_placed": info.jobs_placed if hasattr(info, 'jobs_placed') else 0,
            "jobs_placed_ratio": info.jobs_placed_ratio if hasattr(info, 'jobs_placed_ratio') else 0.0,
            "quality_ratio": info.quality_ratio if hasattr(info, 'quality_ratio') else 0.0,
            "deadline_violation_ratio": info.deadline_violation_ratio if hasattr(info, 'deadline_violation_ratio') else 0.0,
            "job_wait_time": list(info.job_wait_time) if hasattr(info, 'job_wait_time') else [],
        }

    def reset(self, sim_id: str, seed: int = None, rl_problem: str = None) -> dict:
        """Reset simulation, returns observation dict."""
        request = pb2.ResetRequest(sim_id=sim_id, seed=seed if seed else 0)
        response = self.stub.reset(request)
        obs = response.observation

        if rl_problem == "vm_management":
            return {
                "observation": {
                    "infr_state": list(obs.infrastructure_observation) if hasattr(obs, 'infrastructure_observation') else [],
                    "job_cores_waiting_state": obs.job_cores_waiting_observation if hasattr(obs, 'job_cores_waiting_observation') else 0,
                },
                "info": self._step_info_to_dict(response.info, rl_problem),
            }
        # job_placement: use flat_infrastructure_observation (field 3) and jobs_waiting_observation (field 4)
        return {
            "observation": {
                "flat_infrastructure_state": list(obs.flat_infrastructure_observation) if hasattr(obs, 'flat_infrastructure_observation') else [],
                "jobs_waiting_state": list(obs.jobs_waiting_observation) if hasattr(obs, 'jobs_waiting_observation') else [],
            },
            "info": self._step_info_to_dict(response.info, rl_problem),
        }

    def step(self, sim_id: str, action, rl_problem: str = None) -> dict:
        """Execute one step."""
        request = pb2.StepRequest(sim_id=sim_id, action=action)
        response = self.stub.step(request)
        obs = response.observation

        if rl_problem == "vm_management":
            return {
                "observation": {
                    "infr_state": list(obs.infrastructure_observation) if hasattr(obs, 'infrastructure_observation') else [],
                    "job_cores_waiting_state": obs.job_cores_waiting_observation if hasattr(obs, 'job_cores_waiting_observation') else 0,
                },
                "reward": response.reward if hasattr(response, 'reward') else 0.0,
                "terminated": response.terminated if hasattr(response, 'terminated') else False,
                "truncated": response.truncated if hasattr(response, 'truncated') else False,
                "info": self._step_info_to_dict(response.info, rl_problem),
            }
        # job_placement
        return {
            "observation": {
                "flat_infrastructure_state": list(obs.flat_infrastructure_observation) if hasattr(obs, 'flat_infrastructure_observation') else [],
                "jobs_waiting_state": list(obs.jobs_waiting_observation) if hasattr(obs, 'jobs_waiting_observation') else [],
            },
            "reward": response.reward if hasattr(response, 'reward') else 0.0,
            "terminated": response.terminated if hasattr(response, 'terminated') else False,
            "truncated": response.truncated if hasattr(response, 'truncated') else False,
            "info": self._step_info_to_dict(response.info, rl_problem),
        }

    def batch_step(self, sim_id: str, actions, rl_problem: str = None) -> list:
        """Batch step for multiple workers. Returns list of result dicts."""
        items = [
            pb2.BatchStepRequest.StepItem(sim_id=sim_id, action=a)
            for a in actions
        ]
        batch_request = pb2.BatchStepRequest(items=items)
        response = self.stub.batchStep(batch_request)

        results = []
        for r in response.results:
            obs = r.observation
            if rl_problem == "vm_management":
                results.append({
                    "observation": {
                        "infr_state": list(obs.infrastructure_observation) if hasattr(obs, 'infrastructure_observation') else [],
                        "job_cores_waiting_state": obs.job_cores_waiting_observation if hasattr(obs, 'job_cores_waiting_observation') else 0,
                    },
                    "reward": r.reward if hasattr(r, 'reward') else 0.0,
                    "terminated": r.terminated if hasattr(r, 'terminated') else False,
                    "truncated": r.truncated if hasattr(r, 'truncated') else False,
                    "info": self._step_info_to_dict(r.info, rl_problem),
                })
            else:
                results.append({
                    "observation": {
                        "flat_infrastructure_state": list(obs.flat_infrastructure_observation) if hasattr(obs, 'flat_infrastructure_observation') else [],
                        "jobs_waiting_state": list(obs.jobs_waiting_observation) if hasattr(obs, 'jobs_waiting_observation') else [],
                    },
                    "reward": r.reward if hasattr(r, 'reward') else 0.0,
                    "terminated": r.terminated if hasattr(r, 'terminated') else False,
                    "truncated": r.truncated if hasattr(r, 'truncated') else False,
                    "info": self._step_info_to_dict(r.info, rl_problem),
                })
        return results

    def close(self, sim_id: str):
        """Close simulation."""
        request = pb2.CloseRequest(sim_id=sim_id)
        self.stub.close(request)

    def close_channel(self):
        """Close the gRPC channel."""
        self.channel.close()

    def ping(self) -> bool:
        """Health check."""
        try:
            request = pb2.PingRequest()
            response = self.stub.ping(request)
            return response.pong
        except grpc.RpcError:
            return False
