#!/usr/bin/env bash

# Weights & Biases Entity and Project configuration
WANDB_ENTITY="university"
WANDB_PROJECT="visual-scaling-drl"

set -euo pipefail

# Auto-generate per-layer dormant neurons plots for different resolutions
# Resolutions mapping: 48=lr, 64=nr, 80=mr, 96=hd, 112=uhd

RESOLUTIONS=(96)
SUFFIXES=(hd)

# Model variants: each tag template is aligned with its displayed name.
# Use {suffix} where the resolution suffix should be inserted/replaced.
MODEL_VARIANT_TAG_TEMPLATES=(
  'ppo_impala_s3_{suffix}'
  'ppo_impala_s3_{suffix}_bottleneck_50'
  'ppo_impala_s3_{suffix}_bottleneck_drq_50'
  'ppo_impoola_s3_{suffix}'
)

MODEL_VARIANT_DISPLAY_NAMES=(
  'Impala'
  'Impala w/ d(z)=50'
  'Impala w/ DrQv2'
  'Impoola'
)

# Layer keys for each model variant above. This is the easiest way to keep the
# displayed name, tag template, and layer key aligned by index.
# The convolutional layer names are the same across variants; only the linear
# layer differs.
MODEL_VARIANT_LAYER_KEYS_IMPALA=(
  # 'dormant_neurons/0_encoder.network.0.conv'
  # 'dormant_neurons/1_encoder.network.0.res_block0.conv0'
  # 'dormant_neurons/2_encoder.network.0.res_block0.conv1'
  # 'dormant_neurons/3_encoder.network.0.res_block1.conv0'
  # 'dormant_neurons/4_encoder.network.0.res_block1.conv1'
  # 'dormant_neurons/5_encoder.network.1.conv'
  # 'dormant_neurons/6_encoder.network.1.res_block0.conv0'
  'dormant_neurons/15_encoder.network.5'
)

MODEL_VARIANT_LAYER_KEYS_IMPALA_BOTTLENECK_50=(
  # 'dormant_neurons/0_encoder.network.0.conv'
  # 'dormant_neurons/1_encoder.network.0.res_block0.conv0'
  # 'dormant_neurons/2_encoder.network.0.res_block0.conv1'
  # 'dormant_neurons/3_encoder.network.0.res_block1.conv0'
  # 'dormant_neurons/4_encoder.network.0.res_block1.conv1'
  # 'dormant_neurons/5_encoder.network.1.conv'
  # 'dormant_neurons/6_encoder.network.1.res_block0.conv0'
  'dormant_neurons/15_encoder.network.5'
)

MODEL_VARIANT_LAYER_KEYS_IMPALA_DRQ_50=(
  # 'dormant_neurons/0_encoder.network.0.conv'
  # 'dormant_neurons/1_encoder.network.0.res_block0.conv0'
  # 'dormant_neurons/2_encoder.network.0.res_block0.conv1'
  # 'dormant_neurons/3_encoder.network.0.res_block1.conv0'
  # 'dormant_neurons/4_encoder.network.0.res_block1.conv1'
  # 'dormant_neurons/5_encoder.network.1.conv'
  # 'dormant_neurons/6_encoder.network.1.res_block0.conv0'
  'dormant_neurons/15_encoder.network.5'
)

MODEL_VARIANT_LAYER_KEYS_IMPOOLA=(
  # 'dormant_neurons/0_encoder.network.0.conv'
  # 'dormant_neurons/1_encoder.network.0.res_block0.conv0'
  # 'dormant_neurons/2_encoder.network.0.res_block0.conv1'
  # 'dormant_neurons/3_encoder.network.0.res_block1.conv0'
  # 'dormant_neurons/4_encoder.network.0.res_block1.conv1'
  # 'dormant_neurons/5_encoder.network.1.conv'
  # 'dormant_neurons/6_encoder.network.1.res_block0.conv0'
  'dormant_neurons/15_encoder.network.6'
)

OUTPUT_DIR="paper_plots/procgen_hd/ppo/tau3/dormant_neurons/dormant_neurons_per_layer_auto"

# Plot label / file stem for this script.
PLOT_LABEL='Linear (Projection)'
PLOT_STEM='dormant_neurons_linear_projection'

# Environment list (same as original script)
ENV_IDS=(bigfish dodgeball caveflyer ninja starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber)

# Common plot options (copied from the original script)
COMMON_OPTS=(
  --no-check-empty-runs
  --pc.ncols 3
  --pc.ncols-legend 2
  --rliable
  --rc.score_normalization_method 100zero
  --rc.aggregate_metrics_plots
  --rc.normalized_score_threshold_min 0.0
  --rc.normalized_score_threshold 0.5
  --rc.combined_figure
  --pc.ylabel "Fraction (Median)"
  --rc.nsubsamples 11
  --rc.aggregate_fig_title "Dormant Neurons"
  --metric_last_n_average_window 1
  --pc.row_split_after 5
  --pc.rm 2.5
)



mkdir -p "${OUTPUT_DIR}"

if [ "${#MODEL_VARIANT_TAG_TEMPLATES[@]}" -ne "${#MODEL_VARIANT_DISPLAY_NAMES[@]}" ]; then
  echo "ERROR: MODEL_VARIANT_TAG_TEMPLATES and MODEL_VARIANT_DISPLAY_NAMES must have the same length" >&2
  exit 1
fi

impala_variant_display_names=()
impoola_variant_display_name=""
safe_metric_name="${PLOT_STEM}"

# Build python command with --filters for each resolution and both models
cmd=(python -m plot_utils.rlops)

# Add all configured model variants whose tag prefix is non-empty.
for variant_idx in "${!MODEL_VARIANT_TAG_TEMPLATES[@]}"; do
  variant_tag_template=${MODEL_VARIANT_TAG_TEMPLATES[$variant_idx]}
  variant_display=${MODEL_VARIANT_DISPLAY_NAMES[$variant_idx]}
  case "$variant_idx" in
    0) variant_layer_key=${MODEL_VARIANT_LAYER_KEYS_IMPALA[0]} ;;
    1) variant_layer_key=${MODEL_VARIANT_LAYER_KEYS_IMPALA_BOTTLENECK_50[0]} ;;
    2) variant_layer_key=${MODEL_VARIANT_LAYER_KEYS_IMPALA_DRQ_50[0]} ;;
    3) variant_layer_key=${MODEL_VARIANT_LAYER_KEYS_IMPOOLA[0]} ;;
    *) echo "ERROR: unexpected variant index $variant_idx" >&2; exit 1 ;;
  esac

  if [ -z "${variant_tag_template:-}" ]; then
    continue
  fi

  is_impoola=0
  if [ "$variant_idx" -ne 0 ]; then
    is_impoola=1
  fi

  for i in "${!RESOLUTIONS[@]}"; do
    res=${RESOLUTIONS[$i]}
    suf=${SUFFIXES[$i]}

    variant_tag=${variant_tag_template//\{suffix\}/${suf}}
    run_spec="ppo_training?tag=${variant_tag}&seed=1&seed=2&seed=3&seed=4&seed=5&cl=(${res},${res})×${variant_display}"

    if [ "$is_impoola" -eq 1 ]; then
      impoola_variant_display_name=${variant_display}
    else
      impala_variant_display_names+=("${variant_display}")
    fi

    cmd+=(--filters "?we=tumwcps&wpn=impoola-hd&ceik=env_id&cen=exp_name&metric=${variant_layer_key}")
    cmd+=("${run_spec}")
  done
done

# Append --env-ids followed by each env as separate args
cmd+=(--env-ids)
for e in "${ENV_IDS[@]}"; do
  cmd+=("${e}")
done

# Append common options to command
for opt in "${COMMON_OPTS[@]}"; do
  cmd+=("${opt}")
done

# Output filename per metric
out_file="${OUTPUT_DIR}/${safe_metric_name}"
cmd+=(--output-filename "${out_file}")

# Print and run the command
echo "Running plot for layer: ${PLOT_LABEL} (${impala_variant_display_names[*]}) vs (${impoola_variant_display_name}) -> ${out_file}.pdf"
printf '\nCommand:\n'
printf ' %q' "${cmd[@]}"
printf '\n\n'

# Execute the command
"${cmd[@]}"

echo "Saved plot to ${out_file}.pdf"
echo "-------------------------------"

# End
exit 0

