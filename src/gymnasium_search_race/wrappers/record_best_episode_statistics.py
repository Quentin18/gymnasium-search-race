import math
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium import Env
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType


class RecordBestEpisodeStatistics(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    def __init__(self, env: Env[ObsType, ActType]) -> None:
        super().__init__(env)

        self.episode_returns = 0.0
        self.episode_lengths = 0
        self.episode_actions = []
        self.best_episode_returns = -math.inf
        self.best_episode_lengths = 0
        self.best_episode_actions = None

    def step(
        self,
        action: WrapperActType,
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        self.episode_returns += reward
        self.episode_lengths += 1
        self.episode_actions.append(action)

        if (
            terminated or truncated
        ) and self.episode_returns > self.best_episode_returns:
            self.best_episode_returns = self.episode_returns
            self.best_episode_lengths = self.episode_lengths
            self.best_episode_actions = self.episode_actions.copy()

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        self.episode_returns = 0.0
        self.episode_lengths = 0
        self.episode_actions.clear()

        return obs, info
