#!/usr/bin/env bash
# Procgen-HD Hard Mode Plotting Script (Recreates Figure 6)
# Generates aggregate performance curves and IQM normalized score plots for
# Hard distribution mode (100M steps) across tau=2, tau=3, and high-res StarPilot.

# Weights & Biases Entity and Project configuration
WANDB_ENTITY="university"
WANDB_PROJECT="visual-scaling-drl"


# ==============================================================================
# 1. Hard Level Distribution Mode (Tau = 2)
# ==============================================================================
python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
          'ppo_training?tag=ppo_impala_s2_nr_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
          'ppo_training?tag=ppo_impala_s2_hd_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
          'ppo_training?tag=ppo_impoola_s2_nr_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
          'ppo_training?tag=ppo_impoola_s2_hd_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --env-ids bigfish starpilot dodgeball ninja caveflyer coinrun bossfight maze chaser heist plunder fruitbot jumper climber miner leaper \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 2 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.8 \
    --rc.normalized_score_threshold_min 0.2 \
    --pc.ylabel "Normalized Score" \
    --output-filename paper_plots/procgen_hd/ppo/hard/tau2/testing \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Hard Generalization" \
    --metric_last_n_average_window 1 \
    --pc.rm 2.5 \
    --rc.overlay_train_test \
    --pc.row_split_after 2

# ==============================================================================
# 2. Hard Level Distribution Mode (Tau = 3)
# ==============================================================================
python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
          'ppo_training?tag=ppo_impala_s3_nr_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
          'ppo_training?tag=ppo_impala_s3_hd_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
          'ppo_training?tag=ppo_impoola_s3_nr_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
          'ppo_training?tag=ppo_impoola_s3_hd_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --env-ids bigfish starpilot dodgeball ninja caveflyer coinrun bossfight maze chaser heist plunder fruitbot jumper climber miner leaper \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 2 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.8 \
    --rc.normalized_score_threshold_min 0.2 \
    --pc.ylabel "Normalized Score" \
    --output-filename paper_plots/procgen_hd/ppo/hard/tau3/testing \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Hard Generalization" \
    --metric_last_n_average_window 1 \
    --pc.rm 2.5 \
    --rc.overlay_train_test \
    --pc.row_split_after 2

# ==============================================================================
# 3. Extreme High Resolution StarPilot Scaling (128x128 to 224x224)
# ==============================================================================
python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
          'ppo_training?tag=ppo_impoola_s2_nr_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
          'ppo_training?tag=ppo_impoola_s2_hd_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
          'ppo_training?tag=ppo_impoola_s2_128_uhd_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(128,128)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
          'ppo_training?tag=ppo_impoola_s2_160_uhd_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(160,160)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
          'ppo_training?tag=ppo_impoola_s2_192_uhd_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(192,192)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
          'ppo_training?tag=ppo_impoola_s2_224_uhd_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(224,224)$\times$Impoola' \
    --env-ids starpilot \
    --no-check-empty-runs \
    --pc.ncols 1 \
    --pc.ncols-legend 3 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.8 \
    --rc.normalized_score_threshold_min 0.2 \
    --pc.ylabel "Normalized Score" \
    --output-filename paper_plots/procgen_hd/ppo/hard/tau2/high_res_scaling/testing \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Hard Generalization (Starpilot)" \
    --metric_last_n_average_window 1 \
    --pc.rm 2.5 \
    --pc.row_split_after 2
