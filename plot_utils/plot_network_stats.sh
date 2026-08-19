#!/usr/bin/env bash
# Weights & Biases Entity and Project configuration
WANDB_ENTITY="university"
WANDB_PROJECT="visual-scaling-drl"

python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_moe_s3_lr&seed=1&cl=MoE ($\tau=3$) LR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_moe_s3_nr&seed=1&cl=MoE ($\tau=3$) NR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_moe_s3_mr&seed=1&cl=MoE ($\tau=3$) MR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_moe_s3_hd&seed=1&cl=MoE ($\tau=3$) HD' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
             'ppo_training?tag=ppo_moe_s3_uhd&seed=1&cl=MoE ($\tau=3$) UHD' \
    --env-ids bigfish \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 2 \
    --metric_last_n_average_window 1 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.normalized_score_threshold 1 \
    --pc.ylabel "Total Network Parameter" \
    --output-filename paper_plots/results/ppo/stats/tau3_moe/params \
    --rc.nsubsamples 2000 \
    --rc.no_combined_figure

exit 0

python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impala_s2_lr&seed=1&cl=Impala ($\tau=2$) LR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impala_s2_nr&seed=1&cl=Impala ($\tau=2$) NR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impala_s2_mr&seed=1&cl=Impala ($\tau=2$) MR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impala_s2_hd&seed=1&cl=Impala ($\tau=2$) HD' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impala_s2_uhd&seed=1&cl=Impala ($\tau=2$) UHD' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impoola_s2_lr&seed=1&cl=Impoola ($\tau=2$) LR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impoola_s2_nr&seed=1&cl=Impoola ($\tau=2$) NR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impoola_s2_mr&seed=1&cl=Impoola ($\tau=2$) MR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impoola_s2_hd&seed=1&cl=Impoola ($\tau=2$) HD' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impoola_s2_uhd&seed=1&cl=Impoola ($\tau=2$) UHD' \
    --env-ids bigfish \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 2 \
    --metric_last_n_average_window 1 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.normalized_score_threshold 1 \
    --pc.ylabel "Total Network Parameter" \
    --output-filename paper_plots/results/ppo/stats/tau2/params \
    --rc.nsubsamples 2000 \
    --rc.no_combined_figure

python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impala_s3_lr&seed=1&cl=Impala ($\tau=3$) LR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impala_s3_nr&seed=1&cl=Impala ($\tau=3$) NR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impala_s3_mr&seed=1&cl=Impala ($\tau=3$) MR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impala_s3_hd&seed=1&cl=Impala ($\tau=3$) HD' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
          'ppo_training?tag=ppo_impala_s3_uhd&seed=1&cl=Impala ($\tau=3$) UHD' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impoola_s3_lr&seed=1&cl=Impoola ($\tau=3$) LR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impoola_s3_nr&seed=1&cl=Impoola ($\tau=3$) NR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impoola_s3_mr&seed=1&cl=Impoola ($\tau=3$) MR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impoola_s3_hd&seed=1&cl=Impoola ($\tau=3$) HD' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
              'ppo_training?tag=ppo_impoola_s3_uhd&seed=1&cl=Impoola ($\tau=3$) UHD' \
    --env-ids bigfish \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 2 \
    --metric_last_n_average_window 1 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.normalized_score_threshold 1 \
    --pc.ylabel "Total Network Parameter" \
    --output-filename paper_plots/results/ppo/stats/tau4/params \
    --rc.nsubsamples 2000 \
    --rc.no_combined_figure

python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
          'ppo_training?tag=ppo_impala_s4_lr&seed=1&cl=Impala ($\tau=4$) LR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
          'ppo_training?tag=ppo_impala_s4_nr&seed=1&cl=Impala ($\tau=4$) NR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
          'ppo_training?tag=ppo_impala_s4_mr&seed=1&cl=Impala ($\tau=4$) MR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
          'ppo_training?tag=ppo_impala_s4_hd&seed=1&cl=Impala ($\tau=4$) HD' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
          'ppo_training?tag=ppo_impoola_s4_lr&seed=1&cl=Impoola ($\tau=4$) LR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
          'ppo_training?tag=ppo_impoola_s4_nr&seed=1&cl=Impoola ($\tau=4$) NR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
          'ppo_training?tag=ppo_impoola_s4_mr&seed=1&cl=Impoola ($\tau=4$) MR' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
          'ppo_training?tag=ppo_impoola_s4_hd&seed=1&cl=Impoola ($\tau=4$) HD' \
    --env-ids bigfish \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 2 \
    --metric_last_n_average_window 1 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.normalized_score_threshold 1 \
    --pc.ylabel "Total Network Parameter" \
    --output-filename paper_plots/results/ppo/stats/tau4/params \
    --rc.no_combined_figure

#python -m plot_utils.rlops \
#    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_m_macs" \
#              'ppo_training?tag=ppo_impala_s1_random&seed=1&cl=Impala ($\tau=1$)' \
#    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_m_macs" \
#              'ppo_training?tag=ppo_impala_s2_random&seed=1&cl=Impala ($\tau=2$)' \
#    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_m_macs" \
#              'ppo_training?tag=ppo_impala_s3_random&seed=1&cl=Impala ($\tau=3$)' \
#    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_m_macs" \
#              'ppo_training?tag=ppo_impoola_s1_random&seed=1&cl=Impoola ($\tau=1$)' \
#    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_m_macs" \
#              'ppo_training?tag=ppo_impoola_s3_random&seed=1&cl=Impoola ($\tau=3$)' \
#    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_m_macs" \
#              'ppo_training?tag=ppo_impoola_s4_random&seed=1&cl=Impoola ($\tau=4$)' \
#    --env-ids bigfish \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 2 \
#    --metric_last_n_average_window 1 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.normalized_score_threshold 1000000 \
#    --rc.combined_figure \
#    --pc.ylabel "Total Network Parameter" \
#    --output-filename plotting/results/ppo/stats/m_macs \
#    --rc.nsubsamples 2000
#
#python -m plot_utils.rlops \
#    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_param_bytes" \
#              'ppo_training?tag=ppo_impala_s1_random&seed=1&cl=Impala ($\tau=1$)' \
#    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_param_bytes" \
#              'ppo_training?tag=ppo_impala_s2_random&seed=1&cl=Impala ($\tau=2$)' \
#    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_param_bytes" \
#              'ppo_training?tag=ppo_impala_s3_random&seed=1&cl=Impala ($\tau=3$)' \
#    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_param_bytes" \
#              'ppo_training?tag=ppo_impoola_s1_random&seed=1&cl=Impoola ($\tau=1$)' \
#    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_param_bytes" \
#              'ppo_training?tag=ppo_impoola_s3_random&seed=1&cl=Impoola ($\tau=3$)' \
#    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_param_bytes" \
#              'ppo_training?tag=ppo_impoola_s4_random&seed=1&cl=Impoola ($\tau=4$)' \
#    --env-ids bigfish \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 2 \
#    --metric_last_n_average_window 1 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.normalized_score_threshold 1000000 \
#    --rc.combined_figure \
#    --pc.ylabel "Total Network Parameter" \
#    --output-filename plotting/results/ppo/stats/bytes \
#    --rc.nsubsamples 2000
#
#for file in ./plotting/results/ppo/stats/*.pdf; do pdfcrop $file $file;  done

#--filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
#          'ppo_procgen?tag=ppo_group_s3_p80_less_steps&seed=1&cl=Group-Structured $\zeta_F=0.8$ (10 steps)' \

#python -m plot_utils.rlops \
#    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
#              'dqn_procgen?tag=dqn_unstructured_s3_p80&seed=1&cl=Unstructured $\zeta_F=0.8$' \
#    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=charts/total_network_params" \
#              'dqn_procgen?tag=dqn_group_s3_p80_no_all&seed=1&cl=Group-Structured $\zeta_F=0.8$' \
#    --env-ids bigfish \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 2 \
#    --metric_last_n_average_window 1 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.normalized_score_threshold 1000000 \
#    --rc.combined_figure \
#    --pc.ylabel "Total Network Parameter" \
#    --output-filename plotting/results/dqn/params/params \
#    --rc.nsubsamples 2000
#
#for file in ./plotting/results/dqn/params/*.pdf; do pdfcrop $file $file;  done