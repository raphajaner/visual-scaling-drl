#!/usr/bin/env bash
# Procgen-HD Easy Mode Plotting Script (Recreates Figures 1, 4, 5)
# Generates aggregate performance curves and IQM normalized score plots for
# scale factors tau=2, tau=3, and tau=4 across observation resolutions (48x48 to 112x112).

# Weights & Biases Entity and Project configuration
WANDB_ENTITY="university"
WANDB_PROJECT="visual-scaling-drl"


# ==============================================================================
# 1. Scale = 2 (Width Scale tau = 2)
# ==============================================================================
python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s2_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s2_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s2_mr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s2_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s2_uhd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s2_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s2_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s2_mr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s2_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s2_uhd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impoola' \
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
    --output-filename paper_plots/procgen_hd/ppo/tau2/testing \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Generalization" \
    --metric_last_n_average_window 1 \
    --pc.rm 2.5 \
    --rc.overlay_train_test \
    --pc.row_split_after 5

# ==============================================================================
# 2. Scale = 3 (Width Scale tau = 3)
# ==============================================================================
python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s3_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s3_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s3_mr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s3_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s3_uhd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s3_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s3_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s3_mr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s3_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s3_uhd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impoola' \
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
    --output-filename paper_plots/procgen_hd/ppo/tau3/testing \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Generalization" \
    --metric_last_n_average_window 1 \
    --pc.rm 2.5 \
    --rc.overlay_train_test \
    --pc.row_split_after 5

# ==============================================================================
# 3. Scale = 4 (Width Scale tau = 4)
# ==============================================================================
python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s4_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s4_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s4_mr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s4_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impala_s4_uhd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s4_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s4_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s4_mr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s4_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test" \
              'ppo_training?tag=ppo_impoola_s4_uhd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impoola' \
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
    --output-filename paper_plots/procgen_hd/ppo/tau4/testing \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Generalization" \
    --metric_last_n_average_window 1 \
    --pc.rm 2.5 \
    --rc.overlay_train_test \
    --pc.row_split_after 5
