#!/usr/bin/env bash
# GPU/CPU Runtime & Inference Efficiency Profiling Benchmark Suite
# Measures per-step execution timing, forward/backward pass speed, and memory usage
# across observation resolutions (48x48 to 112x112) for Scale=2 and Scale=3.

num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (Total GPUs: $num_gpus)"

# ==============================================================================
# 1. Scale = 2 Timing Benchmarks
# ==============================================================================

# Low Resolution (48x48)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impala_s2_lr OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 48 48 --encoder_type=impala --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impoola_s2_lr OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 48 48 --encoder_type=impoola --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

# Native Resolution (64x64)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impala_s2_nr OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 64 64 --encoder_type=impala --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impoola_s2_nr OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 64 64 --encoder_type=impoola --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

# Medium Resolution (80x80)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impala_s2_mr OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 80 80 --encoder_type=impala --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impoola_s2_mr OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 80 80 --encoder_type=impoola --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

# High Definition (96x96)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impala_s2_hd OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 96 96 --encoder_type=impala --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impoola_s2_hd OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 96 96 --encoder_type=impoola --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

# Ultra High Definition (112x112)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impala_s2_uhd OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 112 112 --encoder_type=impala --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impoola_s2_uhd OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --obs_res 112 112 --encoder_type=impoola --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot


# ==============================================================================
# 2. Scale = 3 Timing Benchmarks
# ==============================================================================

# Low Resolution (48x48)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impala_s3_lr OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 48 48 --encoder_type=impala --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impoola_s3_lr OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 48 48 --encoder_type=impoola --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

# Native Resolution (64x64)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impala_s3_nr OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 64 64 --encoder_type=impala --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impoola_s3_nr OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 64 64 --encoder_type=impoola --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

# Medium Resolution (80x80)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impala_s3_mr OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 80 80 --encoder_type=impala --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impoola_s3_mr OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 80 80 --encoder_type=impoola --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

# High Definition (96x96)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impala_s3_hd OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 96 96 --encoder_type=impala --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impoola_s3_hd OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 96 96 --encoder_type=impoola --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

# Ultra High Definition (112x112)
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impala_s3_uhd OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 112 112 --encoder_type=impala --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_timing_a100_impoola_s3_uhd OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --obs_res 112 112 --encoder_type=impoola --eval_freq=0" \
    --start_seed 1 --num-seeds 3 --workers $num_gpus --no-auto-tag --wandb_project_name visual-scaling-drl --env-ids starpilot
