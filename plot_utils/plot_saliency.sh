#!/usr/bin/env bash
# Tau=2
## Value
### Training

# Weights & Biases Entity and Project configuration
WANDB_ENTITY="university"
WANDB_PROJECT="visual-scaling-drl"

python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/value_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_lr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/value_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_nr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/value_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_mr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/value_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_hd_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/value_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_uhd_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/value_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_lr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/value_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_nr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/value_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_mr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/value_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_hd_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/value_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_uhd_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impoola' \
    --env-ids starpilot \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 4 \
    --rliable \
    --rc.aggregate_metrics_plots \
    --rc.score_normalization_method none \
    --rc.combined_figure \
    --pc.ylabel "Value Mask Sparsity" \
    --output-filename paper_plots/sparsity_mask/ppo/hd/tau2/train/value_mask_sparsity \
    --rc.nsubsamples 11 \
    --rc.aggregate_fig_title "Generalization (\textit{training})" \
    --metric_last_n_average_window 1 \
    --pc.row_split_after 4 \
    --rc.aggregate_fn "mean" \

### Testing
python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/value_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_lr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/value_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_nr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/value_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_mr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/value_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_hd_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/value_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_uhd_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/value_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_lr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/value_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_nr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/value_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_mr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/value_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_hd_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/value_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_uhd_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impoola' \
    --env-ids starpilot \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 4 \
    --rliable \
    --rc.aggregate_metrics_plots \
    --rc.score_normalization_method none \
    --rc.combined_figure \
    --pc.ylabel "Value Mask Sparsity" \
    --output-filename paper_plots/sparsity_mask/ppo/hd/tau2/test/value_mask_sparsity \
    --rc.nsubsamples 11 \
    --rc.aggregate_fig_title "Generalization (\textit{training})" \
    --metric_last_n_average_window 1 \
    --pc.row_split_after 4 \
    --rc.aggregate_fn "mean" \
    --pc.rm 2.5 \

## Policy
### Training
python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/value_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_lr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/value_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_nr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_mr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_hd_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_uhd_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_lr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_nr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_mr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_hd_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/train/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_uhd_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impoola' \
    --env-ids starpilot \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 4 \
    --rliable \
    --rc.aggregate_metrics_plots \
    --rc.score_normalization_method none \
    --rc.combined_figure \
    --pc.ylabel "Policy Mask Sparsity" \
    --output-filename paper_plots/sparsity_mask/ppo/hd/tau2/train/policy_mask_sparsity \
    --rc.nsubsamples 11 \
    --rc.aggregate_fig_title "Generalization (\textit{training})" \
    --metric_last_n_average_window 1 \
    --pc.row_split_after 4 \
    --rc.aggregate_fn "mean" \
    --pc.rm 2.5 \

### Testing
python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_lr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_nr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_mr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_hd_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impala_s2_uhd_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_lr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_nr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_mr_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_hd_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=gradient_vis/test/policy_mask_sparsity" \
          'ppo_training?tag=ppo_impoola_s2_uhd_vis_res&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impoola' \
    --env-ids starpilot \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 4 \
    --rliable \
    --rc.aggregate_metrics_plots \
    --rc.score_normalization_method none \
    --rc.combined_figure \
    --pc.ylabel "Policy Mask Sparsity" \
    --output-filename paper_plots/sparsity_mask/ppo/hd/tau2/test/policy_mask_sparsity \
    --rc.nsubsamples 11 \
    --rc.aggregate_fig_title "Generalization (\textit{training})" \
    --metric_last_n_average_window 1 \
    --pc.row_split_after 4 \
    --rc.aggregate_fn "mean" \
    --pc.rm 2.5 \
