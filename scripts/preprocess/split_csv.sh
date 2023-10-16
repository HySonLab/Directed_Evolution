#!/bin/bash

# Parse command line arguments or set default values
input_file="$1"
train_ratio="${2:-0.8}"
valid_ratio="${3:-0.2}"

# Check if the input file is provided
if [ -z "$input_file" ]; then
  echo "Usage: $0 <input_file> [train_ratio] [valid_ratio]"
  exit 1
fi

# Generate random temporary file names
temp_file1=$(mktemp)

# Get the header from the input file
header=$(head -n 1 "$input_file")

# Shuffle the input file
shuf "$input_file" > "$temp_file1"

# Calculate the number of lines for each split, excluding the header
total_lines=$(($(wc -l < "$temp_file1") - 1))
train_lines=$(awk "BEGIN {print int($train_ratio * $total_lines)}")
valid_lines=$((total_lines - train_lines))

# Split the shuffled data into two files with headers
(head -n 1 "$temp_file1"; head -n "$train_lines" <(tail -n +2 "$temp_file1")) > "${input_file%.*}_train.csv"
(head -n 1 "$temp_file1"; tail -n "$valid_lines" <(tail -n +2 "$temp_file1")) > "${input_file%.*}_valid.csv"

# Clean up temporary files
rm "$temp_file1"

echo "Shuffled and split completed."
echo "Training data: ${input_file%.*}_train.csv,"
echo "Validation data: ${input_file%.*}_valid.csv"
