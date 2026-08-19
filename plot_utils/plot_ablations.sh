#!/usr/bin/env bash
# Procgen-HD Architectural Ablations Plotting Script (Recreates Paper Ablation Figures)
# Generates performance plots comparing Impala and Impoola architectural modifications:
# 1. 5x5 Conv Kernel Receptive Field
# 2. Linear feature bottleneck dimension 50
# 3. DrQ-v2 style output head
# 4. Initial stride-downsampling & deeper ConvSeq blocks

# Weights & Biases Entity and Project configuration
WANDB_ENTITY="university"
WANDB_PROJECT="visual-scaling-drl"


# ==============================================================================
# 1. Impoola Architecture Ablations (Feature Bottleneck & DrQ-v2 Head)
# ==============================================================================
python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s3_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s3_nr_bottleneck_50&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola w/ d(z)=50' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s3_nr_bottleneck_drq_50&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola w/ DrQ-v2' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s3_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s3_hd_bottleneck_50&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola w/ d(z)=50' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s3_hd_bottleneck_drq_50&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola w/ DrQ-v2' \
    --env-ids bigfish starpilot dodgeball ninja caveflyer coinrun bossfight maze chaser heist plunder fruitbot jumper climber miner leaper \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 5 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 1.0 \
    --rc.normalized_score_threshold_min 0.4 \
    --pc.ylabel "Normalized Score" \
    --output-filename paper_plots/procgen_hd/ppo/tau3/impala_ablation/testing_impoola \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Generalization" \
    --metric_last_n_average_window 1 \
    --pc.rm 2.5 \
    --rc.overlay_train_test \
    --pc.row_split_after 3

# ==============================================================================
# 2. Impala Architecture Ablations (Kernels, Bottlenecks, Downsampling, Depth)
# ==============================================================================
python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s3_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s3_nr_kernel_5_c&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala w/ Kernel (5,5)' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s3_nr_bottleneck_50&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala w/ d(z)=50' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s3_nr_bottleneck_drq_50&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala w/ DrQv2' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s3_nr_strong_downsample_first&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala w/ Downsampling' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s3_nr_deeper&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala w/ 4 ConvSeq' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s3_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s3_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s3_hd_kernel_5_c&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala w/ Kernel (5,5)' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s3_hd_bottleneck_50&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala w/ d(z)=50' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s3_hd_bottleneck_drq_50&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala w/ DrQv2' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s3_hd_strong_downsample_first&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala w/ Downsampling' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s3_hd_deeper&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala w/ 4 ConvSeq' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s3_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --env-ids bigfish starpilot dodgeball ninja caveflyer coinrun bossfight maze chaser heist plunder fruitbot jumper climber miner leaper \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 5 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 1.0 \
    --rc.normalized_score_threshold_min 0.4 \
    --pc.ylabel "Normalized Score" \
    --output-filename paper_plots/procgen_hd/ppo/tau3/impala_ablation/testing_96 \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Generalization" \
    --metric_last_n_average_window 1 \
    --pc.rm 2.5 \
    --rc.overlay_train_test \
    --pc.row_split_after 7