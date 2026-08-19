#!/usr/bin/env bash
# Tau=2

# Weights & Biases Entity and Project configuration
WANDB_ENTITY="university"
WANDB_PROJECT="visual-scaling-drl"

python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impala_s2_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impala_s2_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impala_s2_mr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impala_s2_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impala_s2_uhd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impoola_s2_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impoola_s2_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impoola_s2_mr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impoola_s2_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impoola_s2_uhd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impoola' \
    --env-ids bigfish dodgeball caveflyer ninja starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 5 \
    --rliable \
    --rc.score_normalization_method 100zero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.15 \
    --rc.normalized_score_threshold_min 0.0 \
    --pc.ylabel "Fraction" \
    --output-filename paper_plots/procgen_hd/ppo/tau2/dormant_neurons \
    --rc.nsubsamples 11 \
    --rc.aggregate_fig_title "Dormant Neurons" \
    --metric_last_n_average_window 1 \
    --pc.row_split_after 5 \
    --pc.rm 2.5 \

# Tau=3
python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impala_s3_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impala_s3_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impala_s3_mr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impala_s3_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impala_s3_uhd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impoola_s3_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impoola_s3_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impoola_s3_mr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impoola_s3_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impoola_s3_uhd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impoola' \
    --env-ids bigfish dodgeball caveflyer ninja starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 5 \
    --rliable \
    --rc.score_normalization_method 100zero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.15 \
    --rc.normalized_score_threshold_min 0.0 \
    --pc.ylabel "Dormant Neurons Fraction" \
    --output-filename paper_plots/procgen_hd/ppo/tau3/dormant_neurons \
    --rc.nsubsamples 11 \
    --rc.aggregate_fig_title "Generalization" \
    --metric_last_n_average_window 1 \
    --pc.row_split_after 5 \
    --pc.rm 2.5 \


# Tau=4
python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impala_s4_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impala_s4_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impala_s4_mr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impala_s4_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impala_s4_uhd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impoola_s4_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(48,48)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impoola_s4_nr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impoola_s4_mr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(80,80)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impoola_s4_hd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
              'ppo_training?tag=ppo_impoola_s4_uhd&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(112,112)$\times$Impoola' \
    --env-ids bigfish dodgeball caveflyer ninja starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 5 \
    --rliable \
    --rc.score_normalization_method 100zero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.15 \
    --rc.normalized_score_threshold_min 0.0 \
    --pc.ylabel "Dormant Neurons Fraction" \
    --output-filename paper_plots/procgen_hd/ppo/tau4/dormant_neurons \
    --rc.nsubsamples 11 \
    --rc.aggregate_fig_title "Generalization" \
    --metric_last_n_average_window 1 \
    --pc.row_split_after 5 \
    --pc.rm 2.5 \


for file in ./paper_plots/procgen_hd/ppo/*/dormant_neurons*.pdf; do pdfcrop $file $file;  done

# Hard settings
# Tau=2
python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
          'ppo_training?tag=ppo_impala_s2_nr_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
          'ppo_training?tag=ppo_impala_s2_hd_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
          'ppo_training?tag=ppo_impoola_s2_nr_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
          'ppo_training?tag=ppo_impoola_s2_hd_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --env-ids bigfish dodgeball caveflyer ninja starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 2 \
    --rliable \
    --rc.score_normalization_method 100zero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.25 \
    --rc.normalized_score_threshold_min 0.0 \
    --pc.ylabel "Dormant Neurons Fraction" \
    --output-filename paper_plots/procgen_hd/ppo/hard/tau2/dormant_neurons \
    --rc.nsubsamples 11 \
    --rc.aggregate_fig_title "Generalization" \
    --metric_last_n_average_window 1 \
    --pc.row_split_after 2 \
    --pc.rm 2.5 \


# Tau=3
python -m plot_utils.rlops \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
          'ppo_training?tag=ppo_impala_s3_nr_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
          'ppo_training?tag=ppo_impala_s3_hd_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impala' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
          'ppo_training?tag=ppo_impoola_s3_nr_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(64,64)$\times$Impoola' \
    --filters "?we=${WANDB_ENTITY}&wpn=${WANDB_PROJECT}&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction" \
          'ppo_training?tag=ppo_impoola_s3_hd_hard&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(96,96)$\times$Impoola' \
    --env-ids bigfish dodgeball caveflyer ninja starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 2 \
    --rliable \
    --rc.score_normalization_method 100zero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.25 \
    --rc.normalized_score_threshold_min 0.0 \
    --pc.ylabel "Dormant Neurons Fraction" \
    --output-filename paper_plots/procgen_hd/ppo/hard/tau3/dormant_neurons \
    --rc.nsubsamples 11 \
    --rc.aggregate_fig_title "Generalization" \
    --metric_last_n_average_window 1 \
    --pc.row_split_after 2 \
    --pc.rm 2.5 \


for file in ./paper_plots/procgen_hd/ppo/hard/*/dormant_neurons*.pdf; do pdfcrop $file $file;  done
