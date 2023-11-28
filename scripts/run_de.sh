#!/bin/bash

# Parse command line arguments or set default values
dataset="$1"
wt_seq="$2"
ckpt_path="${3}"
wt_fitness="${7:--100}"
n_steps="${4}"
population=128
k="${8:-1}"
num_toks=1
seed="${5:-0}"
devices="${6:-0}"
version=1
num_proposes_per_var="${9:-4}"

python run_discrete_de.py --wt "$wt_seq" --wt_fitness "$wt_fitness" --n_steps "$((n_steps))" --population "$((population))" --num_proposes_per_var "$((num_proposes_per_var))" --seed "$seed" --rm_dups --predictor_ckpt_path "$ckpt_path" --save_name results_${dataset}_steps${n_steps}_pop${population}_pros${num_proposes_per_var}_seed${seed}_k${k}_num${num_toks}_v${version}.csv --k "$((k))" --num_masked_tokens "$((num_toks))" --verbose --devices "$devices" --population_ratio_per_mask 0.6 0.4