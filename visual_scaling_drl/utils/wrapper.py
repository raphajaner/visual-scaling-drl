"""Environment wrappers for tracking episode statistics and metrics."""

import time
from collections import deque
from typing import Optional

import numpy as np
import gym


class RecordEpisodeStatistics(gym.Wrapper):
    """This wrapper will keep track of single episode rewards and lengths."""

    def __init__(self, env, deque_size=None):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None
        if deque_size is not None:
            print("Warning: RecordEpisodeStatistics wrapper does not support deque_size. Ignoring it.")

    def reset(self, **kwargs):
        observations, _ = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, terminateds, truncateds, infos = super().step(action)
        dones = np.logical_or(terminateds, truncateds)

        self.episode_returns += infos["reward"] if self.atari else rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths

        self.episode_returns *= 1 - infos["terminated"] if self.atari else 1 - terminateds
        self.episode_lengths *= 1 - infos["terminated"] if self.atari else 1 - terminateds
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )
