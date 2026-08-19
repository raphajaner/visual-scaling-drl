#!/usr/bin/env bash
# Procgen-HD Architectural Ablations & Rebuttal Benchmark Suite
# Runs multi-seed, multi-GPU PPO benchmarks for paper ablation studies:
# 1. Receptive field scaling (Kernel Size = 5)
# 2. Linear feature bottleneck dimension (Latent Space Dim = 50)
# 3. Head normalization & early downsampling (DrQ-v2 style, strong_downsample_first)
# 4. Extreme high-resolution visual scaling (128x128 to 256x256)

num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (Total GPUs: $num_gpus)"

ALL_ENVS="bigfish starpilot dodgeball ninja caveflyer coinrun bossfight maze chaser heist plunder fruitbot jumper climber miner leaper"

# ==============================================================================
# 1. Receptive Field Ablations (5x5 Conv Kernels)
# ==============================================================================

# Low Resolution (48x48, Kernel 5)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_lr_kernel_5 OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 48 48 --encoder_type=impala --kernel_size 5" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Native Resolution (64x64, Kernel 5)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_nr_kernel_5 OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 64 64 --encoder_type=impala --kernel_size 5" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Medium Resolution (80x80, Kernel 5)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_mr_kernel_5 OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 80 80 --encoder_type=impala --kernel_size 5" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# High Definition (96x96, Kernel 5)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_hd_kernel_5 OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 96 96 --encoder_type=impala --kernel_size 5" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Ultra High Definition (112x112, Kernel 5)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_uhd_kernel_5 OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 112 112 --encoder_type=impala --kernel_size 5" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS


# ==============================================================================
# 2. Linear Bottleneck Ablations (Latent Space Dim = 50)
# ==============================================================================

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_nr_bottleneck_50 OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 64 64 --encoder_type=impala --latent_space_dim 50" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s3_nr_bottleneck_50 OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 64 64 --encoder_type=impoola --latent_space_dim 50" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_hd_bottleneck_50 OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 96 96 --encoder_type=impala --latent_space_dim 50" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s3_hd_bottleneck_50 OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 96 96 --encoder_type=impoola --latent_space_dim 50" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS


# ==============================================================================
# 3. Head Normalization & Initial Downsampling Ablations
# ==============================================================================

# DrQ-v2 Style Output Head (LayerNorm + Tanh)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_hd_use_drqv2_style OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 96 96 --encoder_type=impala --use_drqv2_style" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Strong Initial Downsampling First Layer
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_hd_strong_downsample_first OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 96 96 --encoder_type=impala --strong_downsample_first" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS


# ==============================================================================
# 4. Extreme High-Resolution Visual Scaling (128x128 to 256x256)
# ==============================================================================

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2_128_uhd_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 128 128 --encoder_type=impoola --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2_160_uhd_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 160 160 --encoder_type=impoola --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2_192_uhd_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 192 192 --encoder_type=impoola --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2_224_uhd_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 224 224 --encoder_type=impoola --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2_256_uhd_hard OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 256 256 --encoder_type=impoola --distribution_mode=hard --total_timesteps=100000000" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot