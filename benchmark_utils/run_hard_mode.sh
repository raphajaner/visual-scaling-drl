#!/usr/bin/env bash
# Procgen-HD Visual Scaling Benchmark Suite (Hard Distribution Mode - 100M timesteps)
# Runs multi-seed, multi-GPU PPO benchmarks across observation resolutions (48x48 to 112x112)
# under the Hard level distribution mode.

num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (Total GPUs: $num_gpus)"

ALL_ENVS="bigfish starpilot dodgeball ninja caveflyer coinrun bossfight maze chaser heist plunder fruitbot jumper climber miner leaper"

# ==============================================================================
# 1. Scale = 2 (Width Scale tau = 2) - Hard Mode
# ==============================================================================

# Low Resolution (48x48)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s2_lr_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 48 48 --encoder_type=impala --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2_lr_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 48 48 --encoder_type=impoola --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Native Resolution (64x64)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s2_nr_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 64 64 --encoder_type=impala --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2_nr_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 64 64 --encoder_type=impoola --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Medium Resolution (80x80)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s2_mr_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 80 80 --encoder_type=impala --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2_mr_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 80 80 --encoder_type=impoola --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# High Definition (96x96)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s2_hd_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 96 96 --encoder_type=impala --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2_hd_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 96 96 --encoder_type=impoola --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Ultra High Definition (112x112)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s2_uhd_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 112 112 --encoder_type=impala --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2_uhd_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 112 112 --encoder_type=impoola --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS


# ==============================================================================
# 2. Scale = 3 (Width Scale tau = 3) - Hard Mode
# ==============================================================================

# Native Resolution (64x64)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_nr_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 64 64 --encoder_type=impala --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s3_nr_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 64 64 --encoder_type=impoola --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# High Definition (96x96)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_hd_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 96 96 --encoder_type=impala --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s3_hd_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 96 96 --encoder_type=impoola --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Ultra High Definition (112x112)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_uhd_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 112 112 --encoder_type=impala --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s3_uhd_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 112 112 --encoder_type=impoola --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS
