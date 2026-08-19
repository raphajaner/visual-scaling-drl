#!/usr/bin/env bash
# Weights & Biases Entity and Project configuration
WANDB_ENTITY="university"
WANDB_PROJECT="visual-scaling-drl"

python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train" \
          'ppo_training?tag=ppo_impala_s3_lr_efficiency&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train" \
          'ppo_training?tag=ppo_impala_s3_nr_efficiency&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train" \
          'ppo_training?tag=ppo_impala_s3_mr_efficiency&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train" \
          'ppo_training?tag=ppo_impala_s3_hd_efficiency&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train" \
          'ppo_training?tag=ppo_impoola_s3_lr_efficiency&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train" \
          'ppo_training?tag=ppo_impoola_s3_nr_efficiency&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train" \
          'ppo_training?tag=ppo_impoola_s3_mr_efficiency&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train" \
          'ppo_training?tag=ppo_impoola_s3_hd_efficiency&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --env-ids bigfish starpilot dodgeball ninja caveflyer coinrun bossfight maze chaser heist plunder fruitbot jumper climber miner leaper \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 4 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.9 \
    --rc.normalized_score_threshold_min 0.3 \
    --pc.ylabel "Normalized Score" \
    --output-filename paper_plots/procgen_hd/ppo/efficiency/efficiency \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Efficiency" \
    --metric_last_n_average_window 1 \
    --pc.rm 2.5 \
    --pc.row_split_after 4

for file in ./paper_plots/procgen_hd/ppo/efficiency/*.pdf
  do pdfcrop $file $file
done