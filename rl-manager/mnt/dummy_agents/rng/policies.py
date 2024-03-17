import torch as th
from gymnasium import spaces

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import PyTorchObs

class RngPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete
    ) -> None:
        super().__init__(
            observation_space,
            action_space
        )

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        pass