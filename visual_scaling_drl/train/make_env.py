"""Vector environment creation and wrappers for Procgen-HD."""

import random
import gym
import gymnasium
import numpy as np
import cv2
from procgen.env import ToBaselinesVecEnv, ProcgenGym3Env
from gym3 import ViewerWrapper
from visual_scaling_drl.eval.normalized_score_lists import progcen_hns


def make_an_env(args, seed, normalize_reward, obs_res=(64, 64), env_track_setting="generalization",
                full_distribution=False, render=False):
    """Instantiates a vectorized environment given environment ID and configuration parameters."""
    if args.env_id in list(progcen_hns.keys()):
        envs = make_procgen_env(
            args, env_track_setting, full_distribution=full_distribution,
            normalize_reward=normalize_reward, rand_seed=seed, render=render,
            obs_res=obs_res, distribution_mode=args.distribution_mode,
        )
    else:
        raise ValueError(f"Unknown environment: {args.env_id}")
    return envs


class ProcgenToGymNewAPI(gym.Wrapper):
    """Adapter wrapper converting Procgen Gym3 vector environment API to standard Gym."""

    def reset(self, **kwargs):
        return super().reset(**kwargs), {}

    def step(self, action):
        ob, reward, done, info = super().step(action)
        dict_info = {}
        for key in info[0].keys():
            dict_info[key] = [inf[key] for inf in info]
        terminated = done
        truncated = np.full_like(done, False, dtype=bool)
        return ob, reward, terminated, truncated, dict_info


def _make_procgen_env(num_envs, env_id, env_track, num_levels, rand_seed, obs_res=(64, 64), render=False,
                      distribution_mode="easy"):
    """Creates a Procgen Gym3 vector environment with specified parameters."""

    envs = ProcgenGym3Env(
        num=num_envs,
        env_name=env_id,
        obs_height=obs_res[0],
        obs_width=obs_res[1],
        num_levels=num_levels,
        start_level=(201 if distribution_mode == "easy" else 501) if env_track == "fine_tuning" else 0,
        distribution_mode=distribution_mode,
        rand_seed=rand_seed,
        render_mode="rgb_array" if render else None,
    )

    if render:
        envs = ViewerWrapper(envs, info_key="rgb")

    envs = ProcgenToGymNewAPI(ToBaselinesVecEnv(envs))
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"].transpose((0, 3, 1, 2)))
    envs.single_action_space = envs.action_space

    # (H, W, C) to (C, H, W)
    shape = np.array(envs.observation_space['rgb'].shape)[[2, 0, 1]]

    envs.observation_space["rgb"] = type(envs.observation_space["rgb"])(low=0, high=255, shape=shape, dtype=np.uint8)
    envs.single_observation_space = type(envs.observation_space["rgb"])(low=0, high=255, shape=shape, dtype=np.uint8)
    envs.single_observation_space_gymnasium = gymnasium.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    envs.single_action_space_gymnasium = gymnasium.spaces.Discrete(envs.single_action_space.n)
    envs.is_vector_env = True
    envs.env_type = 'procgen'
    return envs


def make_procgen_env(args, env_track, full_distribution=False, normalize_reward=False, rand_seed=None, obs_res=(64, 64),
                     render=False, distribution_mode="easy"):
    """Creates a Procgen environment with specified parameters and wraps it for Gym compatibility."""

    if env_track == "generalization":
        num_levels = 200 if distribution_mode == "easy" else 500
    elif env_track == "efficiency":
        num_levels = 0
    elif env_track == "fine_tuning":
        num_levels = 100
    else:
        raise ValueError(f"Invalid env_track: {env_track}")

    if full_distribution and env_track == "fine_tuning":
        raise ValueError("Cannot use full distribution in fine tuning track")

    num_levels = 0 if full_distribution else num_levels  # E.g., for evaluation purposes, using ALL possible levels
    envs = _make_procgen_env(args.num_envs, args.env_id, env_track, num_levels, rand_seed, obs_res, render,
                             distribution_mode)

    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    if args.capture_video:
        run_name = f"{args.env_id}_{args.seed}"
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")

    if normalize_reward:
        envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
        envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    return envs
