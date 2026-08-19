#!/usr/bin/env bash
# Procgen-HD Visual Scaling Benchmark Suite (Easy Distribution Mode)
# Runs multi-seed, multi-GPU PPO benchmarks across observation resolutions (48x48 to 112x112)
# and channel width scaling factors (scale=2, 3, 4) for standard Impala and Impoola architectures.

num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (Total GPUs: $num_gpus)"

ALL_ENVS="bigfish starpilot dodgeball ninja caveflyer coinrun bossfight maze chaser heist plunder fruitbot jumper climber miner leaper"

# ==============================================================================
# 1. Scale = 2 (Width Scale tau = 2)
# ==============================================================================

# Low Resolution (48x48)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s2_lr OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 48 48 --encoder_type=impala" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2_lr OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 48 48 --encoder_type=impoola" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Native Resolution (64x64)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s2_nr OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 64 64 --encoder_type=impala" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2_nr OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 64 64 --encoder_type=impoola" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Medium Resolution (80x80)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s2_mr OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 80 80 --encoder_type=impala" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2_mr OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 80 80 --encoder_type=impoola" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# High Definition (96x96)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s2_hd OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 96 96 --encoder_type=impala" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2_hd OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 96 96 --encoder_type=impoola" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Ultra High Definition (112x112)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s2_uhd OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 112 112 --encoder_type=impala" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2_uhd OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 112 112 --encoder_type=impoola" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS


# ==============================================================================
# 2. Scale = 3 (Width Scale tau = 3)
# ==============================================================================

# Low Resolution (48x48)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_lr OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 48 48 --encoder_type=impala" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s3_lr OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 48 48 --encoder_type=impoola" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Native Resolution (64x64)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_nr OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 64 64 --encoder_type=impala" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s3_nr OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 64 64 --encoder_type=impoola" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Medium Resolution (80x80)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_mr OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 80 80 --encoder_type=impala" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s3_mr OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 80 80 --encoder_type=impoola" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# High Definition (96x96)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_hd OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 96 96 --encoder_type=impala" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s3_hd OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 96 96 --encoder_type=impoola" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Ultra High Definition (112x112)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3_uhd OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 112 112 --encoder_type=impala" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s3_uhd OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 112 112 --encoder_type=impoola" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS


# ==============================================================================
# 3. Scale = 4 (Width Scale tau = 4)
# ==============================================================================

# Low Resolution (48x48)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s4_lr OMP_NUM_THREADS=1 python ppo_training.py --scale=4 --obs_res 48 48 --encoder_type=impala" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s4_lr OMP_NUM_THREADS=1 python ppo_training.py --scale=4 --obs_res 48 48 --encoder_type=impoola" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Native Resolution (64x64)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s4_nr OMP_NUM_THREADS=1 python ppo_training.py --scale=4 --obs_res 64 64 --encoder_type=impala" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s4_nr OMP_NUM_THREADS=1 python ppo_training.py --scale=4 --obs_res 64 64 --encoder_type=impoola" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Medium Resolution (80x80)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s4_mr OMP_NUM_THREADS=1 python ppo_training.py --scale=4 --obs_res 80 80 --encoder_type=impala" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s4_mr OMP_NUM_THREADS=1 python ppo_training.py --scale=4 --obs_res 80 80 --encoder_type=impoola" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# High Definition (96x96)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s4_hd OMP_NUM_THREADS=1 python ppo_training.py --scale=4 --obs_res 96 96 --encoder_type=impala" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s4_hd OMP_NUM_THREADS=1 python ppo_training.py --scale=4 --obs_res 96 96 --encoder_type=impoola" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

# Ultra High Definition (112x112)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s4_uhd OMP_NUM_THREADS=1 python ppo_training.py --scale=4 --obs_res 112 112 --encoder_type=impala" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s4_uhd OMP_NUM_THREADS=1 python ppo_training.py --scale=4 --obs_res 112 112 --encoder_type=impoola" \
    --start_seed 1 --num-seeds 5 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids $ALL_ENVS