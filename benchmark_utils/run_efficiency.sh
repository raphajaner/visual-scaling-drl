#!/usr/bin/env bash
# Procgen-HD Visual Scaling Benchmark Suite (Efficiency Track - Unlimited Level Distribution)
# Trains PPO agents on unlimited level distribution (env_track_setting=efficiency)
# across resolutions (48x48 to 96x96) for Scale=3 Impala and Impoola architectures.

num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (Total GPUs: $num_gpus)"

ALL_ENVS="bigfish starpilot dodgeball ninja caveflyer coinrun bossfight maze chaser heist plunder fruitbot jumper climber miner leaper"

# ==============================================================================
# Efficiency Track (env_track_setting = efficiency, num_levels = 0)
# ==============================================================================

# Low Resolution (48x48)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_lr_efficiency OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 48 48 --encoder_type=impala --env_track_setting=efficiency" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s3_lr_efficiency OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 48 48 --encoder_type=impoola --env_track_setting=efficiency" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Native Resolution (64x64)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_nr_efficiency OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 64 64 --encoder_type=impala --env_track_setting=efficiency" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s3_nr_efficiency OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 64 64 --encoder_type=impoola --env_track_setting=efficiency" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Medium Resolution (80x80)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_mr_efficiency OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 80 80 --encoder_type=impala --env_track_setting=efficiency" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s3_mr_efficiency OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 80 80 --encoder_type=impoola --env_track_setting=efficiency" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# High Definition (96x96)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_hd_efficiency OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 96 96 --encoder_type=impala --env_track_setting=efficiency" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s3_hd_efficiency OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 96 96 --encoder_type=impoola --env_track_setting=efficiency" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS
