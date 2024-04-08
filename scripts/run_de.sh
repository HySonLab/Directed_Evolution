#!/bin/bash

# Parse command line arguments or set default values
dataset="$1"
n_steps="${2}"
seed="${3:-0}"
devices="${4:-0}"
k="${5:-1}"
num_proposes_per_var="${6:-4}"
num_chunk="${7:-1}"
ckpt_path=$8
population=128
num_toks=1
pretrained_mutation_name="facebook/esm2_t12_35M_UR50D"
model_name="esm2-35M"

python scripts/run_discrete_de.py --task "$dataset" --n_steps "$((n_steps))" --population "$((population))" \
                          --num_proposes_per_var "$((num_proposes_per_var))" --seed "$seed" --rm_dups \
                          --save_name results_${dataset}_model=${model_name}_steps${n_steps}_pop${population}_pros${num_proposes_per_var}_seed${seed}_k${k}_num${num_toks}_imp.csv \
                          --k "$((k))" --num_masked_tokens "$((num_toks))" --verbose --devices "$devices" --predictor_ckpt_path "$ckpt_path" \
                          --population_ratio_per_mask 0.1 0.9 --pretrained_mutation_name "$pretrained_mutation_name"