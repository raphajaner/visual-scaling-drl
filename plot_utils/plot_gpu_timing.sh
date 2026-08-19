#!/usr/bin/env bash
# GPU/CPU Runtime & VRAM Efficiency Plotting Script (Recreates Figure 8 & Table 1)
# Generates GPU wall-clock training time, peak VRAM memory allocation, and per-step timing plots
# across scale factors tau=2 and tau=3 for resolutions (48x48 to 112x112).

# Weights & Biases Entity and Project configuration
WANDB_ENTITY="university"
WANDB_PROJECT="visual-scaling-drl"


# ==============================================================================
# 1. Peak VRAM Allocated (MB) - Tau = 2 & Tau = 3
# ==============================================================================
python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/vram_peak_allocated_mb" \
          'ppo_training?tag=ppo_timing_a100_impala_s2_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/vram_peak_allocated_mb" \
          'ppo_training?tag=ppo_timing_a100_impala_s2_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/vram_peak_allocated_mb" \
          'ppo_training?tag=ppo_timing_a100_impala_s2_mr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/vram_peak_allocated_mb" \
          'ppo_training?tag=ppo_timing_a100_impala_s2_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/vram_peak_allocated_mb" \
          'ppo_training?tag=ppo_timing_a100_impala_s2_uhd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/vram_peak_allocated_mb" \
          'ppo_training?tag=ppo_timing_a100_impoola_s2_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/vram_peak_allocated_mb" \
          'ppo_training?tag=ppo_timing_a100_impoola_s2_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/vram_peak_allocated_mb" \
          'ppo_training?tag=ppo_timing_a100_impoola_s2_mr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/vram_peak_allocated_mb" \
          'ppo_training?tag=ppo_timing_a100_impoola_s2_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/vram_peak_allocated_mb" \
          'ppo_training?tag=ppo_timing_a100_impoola_s2_uhd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impoola' \
    --env-ids starpilot \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 4 \
    --rliable \
    --rc.aggregate_metrics_plots \
    --rc.score_normalization_method none \
    --rc.combined_figure \
    --pc.ylabel "Peak VRAM Allocated (MB)" \
    --output-filename paper_plots/elapsed_train_time/ppo/tau2/vram_peak_allocated_mb \
    --rc.nsubsamples 11 \
    --rc.aggregate_fig_title "Generalization (\textit{training})" \
    --metric_last_n_average_window 1 \
    --pc.row_split_after 4

# ==============================================================================
# 2. Total Elapsed Training Wall-Clock Time - Tau = 2 & Tau = 3
# ==============================================================================
python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/elapsed_train_time" \
          'ppo_training?tag=ppo_timing_a100_impala_s2_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/elapsed_train_time" \
          'ppo_training?tag=ppo_timing_a100_impala_s2_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/elapsed_train_time" \
          'ppo_training?tag=ppo_timing_a100_impala_s2_mr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/elapsed_train_time" \
          'ppo_training?tag=ppo_timing_a100_impala_s2_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/elapsed_train_time" \
          'ppo_training?tag=ppo_timing_a100_impala_s2_uhd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/elapsed_train_time" \
          'ppo_training?tag=ppo_timing_a100_impoola_s2_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/elapsed_train_time" \
          'ppo_training?tag=ppo_timing_a100_impoola_s2_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/elapsed_train_time" \
          'ppo_training?tag=ppo_timing_a100_impoola_s2_mr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/elapsed_train_time" \
          'ppo_training?tag=ppo_timing_a100_impoola_s2_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/elapsed_train_time" \
          'ppo_training?tag=ppo_timing_a100_impoola_s2_uhd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impoola' \
    --env-ids starpilot \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 4 \
    --rliable \
    --rc.aggregate_metrics_plots \
    --rc.score_normalization_method none \
    --rc.combined_figure \
    --pc.ylabel "Elapsed Training Time" \
    --output-filename paper_plots/elapsed_train_time/ppo/tau2/elapsed_train_time \
    --rc.nsubsamples 11 \
    --rc.aggregate_fig_title "Generalization (\textit{training})" \
    --metric_last_n_average_window 1 \
    --pc.row_split_after 4
