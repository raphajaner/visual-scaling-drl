#!/usr/bin/env bash
# Procgen-HD Example Observation Rendering Suite
# Renders example high-definition observations across resolutions (48x48 to 192x192) for figure generation.

num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (Total GPUs: $num_gpus)"

ALL_ENVS="bigfish starpilot dodgeball ninja caveflyer coinrun bossfight maze chaser heist plunder fruitbot jumper climber miner leaper"

# ==============================================================================
# Rendering Example Observations Across Resolutions
# ==============================================================================

# Low Resolution (48x48)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_example_obs_48 OMP_NUM_THREADS=1 python ppo_training.py --scale=1 --num_envs 32 --obs_res 48 48 --no-track" \
    --start_seed 1 --num-seeds 1 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Native Resolution (64x64)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_example_obs_64 OMP_NUM_THREADS=1 python ppo_training.py --scale=1 --num_envs 32 --obs_res 64 64 --no-track" \
    --start_seed 1 --num-seeds 1 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Medium Resolution (80x80)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_example_obs_80 OMP_NUM_THREADS=1 python ppo_training.py --scale=1 --num_envs 32 --obs_res 80 80 --no-track" \
    --start_seed 1 --num-seeds 1 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# High Definition (96x96)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_example_obs_96 OMP_NUM_THREADS=1 python ppo_training.py --scale=1 --num_envs 32 --obs_res 96 96 --no-track" \
    --start_seed 1 --num-seeds 1 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Ultra High Definition (112x112)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_example_obs_112 OMP_NUM_THREADS=1 python ppo_training.py --scale=1 --num_envs 32 --obs_res 112 112 --no-track" \
    --start_seed 1 --num-seeds 1 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Extreme Resolution (192x192)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_example_obs_192 OMP_NUM_THREADS=1 python ppo_training.py --scale=1 --num_envs 32 --obs_res 192 192 --no-track" \
    --start_seed 1 --num-seeds 1 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS
