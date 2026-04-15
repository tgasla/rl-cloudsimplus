import grpc
from . import cloudsimplus_pb2
from . import cloudsimplus_pb2_grpc
from typing import Dict, Tuple, Any, Optional
import json


class CloudSimGrpcClient:
    """
    Thin gRPC client wrapping the CloudSimService.
    Each client instance connects to ONE gRPC server (one Java JVM).

    When used with SubprocVecEnv, each worker process creates its own
    CloudSimGrpcClient pointing to its own Java subprocess on a dedicated port.
    """

    def __init__(self, host: str = "localhost", port: int = 50051):
        self.host = host
        self.port = port
        self._channel = None
        self._stub = None
        self._connect()

    def _connect(self):
        self._channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        self._stub = cloudsimplus_pb2_grpc.CloudSimServiceStub(self._channel)

    def create_simulation(self, params: Dict, jobs_json: str) -> str:
        """Create a simulation and return its ID."""
        request = cloudsimplus_pb2.CreateRequest(
            params_json=json.dumps(params),
            jobs_json=jobs_json,
        )
        response = self._stub.createSimulation(request)
        return response.sim_id

    def reset(self, sim_id: str, seed: int = 0) -> Tuple[Dict, Dict]:
        """Reset simulation, return (observation, info)."""
        request = cloudsimplus_pb2.ResetRequest(sim_id=sim_id, seed=seed)
        response = self._stub.reset(request)
        obs = self._convert_observation(response.observation)
        info = self._convert_step_info(response.info)
        return obs, info

    def step(self, sim_id: str, action) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step. Returns (obs, reward, terminated, truncated, info)."""
        request = cloudsimplus_pb2.StepRequest(
            sim_id=sim_id,
            action=action,
        )
        response = self._stub.step(request)
        obs = self._convert_observation(response.observation)
        info = self._convert_step_info(response.info)
        return obs, response.reward, response.terminated, response.truncated, info

    def close(self, sim_id: str):
        """Close and clean up a simulation."""
        request = cloudsimplus_pb2.CloseRequest(sim_id=sim_id)
        self._stub.close(request)

    def ping(self) -> bool:
        """Health check."""
        try:
            response = self._stub.ping(cloudsimplus_pb2.PingRequest())
            return response.alive
        except Exception:
            return False

    def batch_step(self, items):
        """
        Execute batched steps for N simulations in one RPC.
        items: list of (sim_id, action) tuples.
        Returns list of (obs, reward, terminated, truncated, info) tuples in same order.
        """
        batch_items = [
            cloudsimplus_pb2.BatchStepRequest.StepItem(sim_id=sid, action=act)
            for sid, act in items
        ]
        request = cloudsimplus_pb2.BatchStepRequest(items=batch_items)
        response = self._stub.batchStep(request)
        results = []
        for r in response.results:
            obs = self._convert_observation(r.observation)
            info = self._convert_step_info(r.info)
            results.append((obs, r.reward, r.terminated, r.truncated, info))
        return results

    def close_channel(self):
        """Close the gRPC channel."""
        if self._channel:
            self._channel.close()

    @staticmethod
    def _convert_observation(grpc_obs) -> Dict:
        return {
            "infr_state": list(grpc_obs.infrastructure_observation),
            "job_cores_waiting_state": grpc_obs.job_cores_waiting_observation,
        }

    @staticmethod
    def _convert_step_info(info) -> Dict:
        return {
            "job_wait_reward": info.job_wait_reward,
            "running_vm_cores_reward": info.running_vm_cores_reward,
            "unutilized_vm_cores_reward": info.unutilized_vm_cores_reward,
            "invalid_reward": info.invalid_reward,
            "isValid": info.is_valid,
            "job_wait_time": list(info.job_wait_time),
            "unutilized_vm_core_ratio": info.unutilized_vm_core_ratio,
            "observation_tree_array": list(info.observation_tree_array),
            "host_affected": info.host_affected,
            "cores_changed": info.cores_changed,
        }
