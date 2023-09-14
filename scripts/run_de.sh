#!/bin/bash

num_processes=$1

python run_discrete_de.py --max_sequence_length 1024 --n_steps 30 --population 10 --num_masked_tokens 1 --verbose --num_processes $num_processes --k 1 --tranception_type Small --mask_high_importance --num_proposes_per_var 5 --fasta_file ../data/part-0.fasta