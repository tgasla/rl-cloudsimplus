from typing import Any, Dict, ClassVar, Tuple, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from dummy_agents.common.no_policy_algorithm import NoPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.utils import is_vectorized_observation

SelfRNG = TypeVar("SelfRNG", bound="RNG")

class RNG(NoPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
    }
    policy: MlpPolicy

    def __init__(
        self,
        policy: Union[str, Type[MlpPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy = policy,
            env = env,
            learning_rate=learning_rate,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        self.policy = policy
        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        super()._setup_model()

    def train(self) -> None:
        """
        Nothing to train.
        """
        pass

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function for random policy.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        
        if self.is_vectorized_observation(observation):
            if isinstance(observation, dict):
                n_batch = observation[next(iter(observation.keys()))].shape[0]
            else:
                n_batch = observation.shape[0]
            action = np.array([self.action_space.sample() for _ in range(n_batch)])
        else:
            action = np.array(self.action_space.sample())
        
        return action, state
    
    def learn(
        self: SelfRNG,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "RNG",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfRNG:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def is_vectorized_observation(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> bool:
        """
        Check whether or not the observation is vectorized,
        apply transposition to image (so that they are channel-first) if needed.
        This is used in DQN when sampling random action (epsilon-greedy policy)

        :param observation: the input observation to check
        :return: whether the given observation is vectorized or not
        """
        vectorized_env = False
        if isinstance(observation, dict):
            assert isinstance(
                self.observation_space, spaces.Dict
            ), f"The observation provided is a dict but the obs space is {self.observation_space}"
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                vectorized_env = vectorized_env or is_vectorized_observation(maybe_transpose(obs, obs_space), obs_space)
        else:
            vectorized_env = is_vectorized_observation(
                maybe_transpose(observation, self.observation_space), self.observation_space
            )
        return vectorized_env