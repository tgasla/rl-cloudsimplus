"""CloudSim gRPC Client wrapper using generated protobuf stubs."""
import grpc
from . import cloudsimplus_pb2 as pb2
from . import cloudsimplus_pb2_grpc as pb2_grpc


class CloudSimGrpcClient:
    """Wrapper around CloudSimServiceStub for CloudSim Plus gRPC communication."""

    def __init__(self, host="localhost", port=50051):
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = pb2_grpc.CloudSimServiceStub(self.channel)

    def create_simulation(self, params_json: str, jobs_json: str) -> str:
        """Create a new simulation, returns sim_id."""
        request = pb2.CreateRequest(params_json=params_json, jobs_json=jobs_json)
        response = self.stub.createSimulation(request)
        return response.sim_id

    def _step_info_to_dict(self, info):
        """Convert StepInfo protobuf to dict."""
        if not info:
            return {}
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

    def reset(self, sim_id: str, seed: int = None) -> dict:
        """Reset simulation, returns observation dict."""
        request = pb2.ResetRequest(sim_id=sim_id, seed=seed if seed else 0)
        response = self.stub.reset(request)
        obs = response.observation
        return {
            "observation": {
                "infr_state": list(obs.infrastructure_observation) if hasattr(obs, 'infrastructure_observation') else [],
                "job_cores_waiting_state": obs.job_cores_waiting_observation if hasattr(obs, 'job_cores_waiting_observation') else 0,
            },
            "info": self._step_info_to_dict(response.info),
        }

    def step(self, sim_id: str, action) -> dict:
        """Execute one step, returns (obs, reward, terminated, truncated, info)."""
        request = pb2.StepRequest(
            sim_id=sim_id,
            action=action,
        )
        response = self.stub.step(request)
        obs = response.observation
        return {
            "observation": {
                "infr_state": list(obs.infrastructure_observation) if hasattr(obs, 'infrastructure_observation') else [],
                "job_cores_waiting_state": obs.job_cores_waiting_observation if hasattr(obs, 'job_cores_waiting_observation') else 0,
            },
            "reward": response.reward if hasattr(response, 'reward') else 0.0,
            "terminated": response.terminated if hasattr(response, 'terminated') else False,
            "truncated": response.truncated if hasattr(response, 'truncated') else False,
            "info": self._step_info_to_dict(response.info),
        }

    def batch_step(self, sim_id: str, actions) -> list:
        """Batch step for multiple actions."""
        items = [
            pb2.BatchStepRequest.StepItem(sim_id=sim_id, action=a)
            for a in actions
        ]
        batch_request = pb2.BatchStepRequest(items=items)
        response = self.stub.batchStep(batch_request)
        return [
            {
                "observation": {
                    "infr_state": list(r.observation.infrastructure_observation) if hasattr(r, 'observation') and hasattr(r.observation, 'infrastructure_observation') else [],
                    "job_cores_waiting_state": r.observation.job_cores_waiting_observation if hasattr(r, 'observation') and hasattr(r.observation, 'job_cores_waiting_observation') else 0,
                },
                "reward": r.reward if hasattr(r, 'reward') else 0.0,
                "terminated": r.terminated if hasattr(r, 'terminated') else False,
                "truncated": r.truncated if hasattr(r, 'truncated') else False,
                "info": self._step_info_to_dict(r.info),
            }
            for r in response.results
        ]

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
