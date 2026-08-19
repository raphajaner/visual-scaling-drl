"""Evaluation protocols and rollouts for generalization tracks and score normalization."""

from collections import deque

import wandb
import numpy as np
import torch

from visual_scaling_drl.train.make_env import make_an_env
from visual_scaling_drl.eval.normalized_score_lists import progcen_hns

from visual_scaling_drl.utils.gradient_saliency import GradientVisualizer


def rollout(envs, agent, n_episodes=10000, noise_scale=None, deterministic=True):
    """Executes evaluation rollouts across parallel environment workers until n_episodes complete."""
    device = next(agent.parameters()).device

    # We cannot simply append eps whenever one is ready because this would bias the evaluation towards eps that are fast
    eval_avg_return = []
    eps_to_do_per_env = np.zeros(envs.num_envs)
    for idx in range(n_episodes):
        eps_to_do_per_env[idx % envs.num_envs] += 1

    assert sum(eps_to_do_per_env) == n_episodes, f"Sum of eps_to_do_per_env is broken: {sum(eps_to_do_per_env)}"

    agent.eval()
    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    obs_shape = next_obs.shape

    with torch.inference_mode():
        while len(eval_avg_return) < n_episodes:
            action = agent.get_action(next_obs, deterministic=deterministic)
            next_obs, _, terminated, truncated, info = envs.step(action.cpu().numpy())

            if noise_scale is not None:
                next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
                noise = torch.randn(obs_shape, device=device) * noise_scale
                next_obs.add_(noise.round()).clamp_(0.0, 255.0)
            else:
                next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)

            if "_episode" in info.keys():
                for i in range(len(info["_episode"])):
                    if info["_episode"][i] and eps_to_do_per_env[i] > 0:
                        eval_avg_return.append(info["episode"]["r"][i])
                        eps_to_do_per_env[i] -= 1

    agent.train()
    return eval_avg_return


def _get_normalized_score(eval_avg_return, game_range):
    """Computes Human-Normalized Score (HNS) given mean return and min/max score bounds."""
    if game_range is not None:
        normalized_score = (np.mean(eval_avg_return) - game_range[1]) / (game_range[2] - game_range[1])
    else:
        normalized_score = np.mean(eval_avg_return)
    return normalized_score


def _get_game_range(env_id):
    """Retrieves score range bounds (random, human, max) for a specific Procgen environment."""
    for game_name, game_range in progcen_hns.items():
        if env_id in game_name:
            print(f"Game range: {game_range}")
            return game_range

    raise ValueError(f"Unknown environment: {env_id}")


def get_normalized_score(env_id, eval_avg_return):
    """Computes normalized score for a named game given evaluation episode returns."""
    game_range = _get_game_range(env_id)
    return _get_normalized_score(eval_avg_return, game_range)


def _evaluate_and_log_results(env_id, eval_avg_return, global_step, prefix, postfix=""):
    """Logs evaluation returns and Human-Normalized Scores to Weights & Biases."""
    normalized_score = get_normalized_score(env_id, eval_avg_return)
    wandb.log({
        f"global_step{postfix}": global_step,
        f"scores{postfix}/normalized_score_{prefix}": normalized_score,
        f"scores{postfix}/eval_avg_return_{prefix}": np.mean(eval_avg_return),
    })
    print(f"\nNormalized score {prefix} ({global_step}): {normalized_score}")


def run_training_track(agent, args, global_step=None, postfix=""):
    """Run evaluation on the training track."""
    envs = make_an_env(args, seed=args.seed, normalize_reward=False, obs_res=args.obs_res,
                       env_track_setting=args.env_track_setting,
                       full_distribution=False) # Note: full_distribution=False for training track

    eval_avg_return = rollout(envs, agent, args.n_episodes_rollout, deterministic=args.deterministic_rollout)
    envs.close()
    _evaluate_and_log_results(args.env_id, eval_avg_return, global_step, "train", postfix)


def run_test_track(agent, args, global_step=None, postfix=""):
    """Run evaluation on the test track, i.e., full distribution of environments."""
    envs = make_an_env(args, seed=args.seed, normalize_reward=False, obs_res=args.obs_res,
                       env_track_setting=args.env_track_setting,
                       full_distribution=True) # Note: full_distribution=True for test track

    eval_avg_return_test = rollout(envs, agent, args.n_episodes_rollout, deterministic=args.deterministic_rollout)
    envs.close()
    _evaluate_and_log_results(args.env_id, eval_avg_return_test, global_step, "test", postfix)


def run_gradient_visualization(
        agent,
        args,
        global_step: int,
        n_episodes: int = 5,
        max_frames_per_episode: int = 500,
        track: str = "both",
        threshold: float = 0.1,
        background_brightness: float = 0.0,
):
    """
    Run gradient visualization separately from main evaluation.
    
    Args:
        agent: PPO agent
        args: Arguments (uses env_id, seed, obs_res, env_track_setting, deterministic_rollout)
        global_step: Current training step for logging
        n_episodes: Number of episodes to visualize (default: 5)
        max_frames_per_episode: Max frames per episode (default: 500)
        track: Which track to visualize - "train", "test", or "both"
        threshold: Direct threshold in [0, 1] for attention masking (default: 0.01)
        background_brightness: Brightness of non-attended regions (default: 0.0 = black)
    """
    from visual_scaling_drl.train.make_env import make_an_env

    print(f"\n[GradientVis] Starting gradient visualization at step {global_step}")

    if track in ["train", "both"]:
        _run_gradient_vis_track(
            agent, args, global_step, n_episodes, max_frames_per_episode,
            full_distribution=False, threshold=threshold,
            background_brightness=background_brightness, prefix="gradient_vis/train"
        )

    if track in ["test", "both"] and args.env_track_setting == "generalization":
        _run_gradient_vis_track(
            agent, args, global_step, n_episodes, max_frames_per_episode,
            full_distribution=True, threshold=threshold,
            background_brightness=background_brightness, prefix="gradient_vis/test"
        )

    print(f"[GradientVis] Done\n")


def _run_gradient_vis_track(
        agent, args, global_step, n_episodes, max_frames_per_episode,
        full_distribution, threshold, background_brightness, prefix
):
    """Run gradient visualization for a single track."""
    from visual_scaling_drl.train.make_env import make_an_env

    device = next(agent.parameters()).device

    envs = make_an_env(
        args, seed=args.seed, normalize_reward=False, obs_res=args.obs_res,
        env_track_setting=args.env_track_setting, full_distribution=full_distribution
    )

    visualizer = GradientVisualizer(
        agent=agent,
        device=device,
        n_episodes=n_episodes,
        max_frames_per_episode=max_frames_per_episode,
        threshold=threshold,
        background_brightness=background_brightness,
    )

    # Track which env we're recording from
    tracking_env_idx = 0

    agent.eval()
    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

    step_count = 0

    while visualizer.is_collecting():
        # Collect gradient from tracked env
        visualizer.step(next_obs[tracking_env_idx], done=False, env_idx=0)

        # Get action
        with torch.inference_mode():
            action = agent.get_action(next_obs, deterministic=args.deterministic_rollout)

        next_obs, _, terminated, truncated, info = envs.step(action.cpu().numpy())
        next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
        step_count += 1

        # Check if tracked env finished
        episode_ended = False
        if envs.env_type == "atari":
            next_done = np.logical_or(terminated, truncated)
            if next_done[tracking_env_idx] and info["lives"][tracking_env_idx] == 0:
                episode_ended = True
        else:
            if "_episode" in info.keys() and info["_episode"][tracking_env_idx]:
                episode_ended = True

        if episode_ended:
            visualizer.step(next_obs[tracking_env_idx], done=True, env_idx=0)
            print(f"[GradientVis] Episode {visualizer.episodes_collected}/{n_episodes} saved "
                  f"({len(visualizer.episodes_data[-1]) if visualizer.episodes_data else 0} frames)")

    agent.train()
    envs.close()

    visualizer.log_to_wandb(global_step, prefix=prefix)
