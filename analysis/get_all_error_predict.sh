#!/bin/bash

# Define the directory to search in and the output file
search_dir="/home/xiachunwei/Projects/alpaca-lora-decompilation/tmp_validate_exebench"
output_file="all_error_predict_list.txt"

# Find all files named "error_predict.error" and save their full paths to the output file
find "$search_dir" -type f -name "error_predict.error" > "$output_file"

# Print a message indicating the task is done
echo "File paths have been saved to $output_file"
