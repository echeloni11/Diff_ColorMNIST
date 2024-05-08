#!/bin/bash

# Initial experiment number
num=31
max_experiments=40 # maximum number of experiments to run

# Define arrays of parameters
declare -a cond_modes=("AdaGN")
declare -a classifier_names=("0" "0.01" "0.05" "0.1" "1")
declare -a class_types=("logit" "label")

# Directory to store status files
status_dir="./experiment_status"
mkdir -p "$status_dir"

# Loop over each combination of parameters
for cond_mode in "${cond_modes[@]}"; do
    for classifier_name in "${classifier_names[@]}"; do
        for class_type in "${class_types[@]}"; do
            # Construct the date string
            date="240508_$num"
            
            # Increment the experiment number
            ((num++))
            
            # # Skip the first experiment
            # if [ "$num" -eq 2 ]; then
            #     continue
            # fi

            status_file="${status_dir}/${date}_${cond_mode}_${classifier_name}_${class_type}.status"

            # Skip if the status file exists and indicates completion
            if [[ -f "$status_file" && $(cat "$status_file") == "SUCCESS" ]]; then
                echo "Skipping completed experiment: $date $cond_mode $classifier_name $class_type"
                continue
            fi

            # Construct command
            command="python script.py --date $date --cond_mode $cond_mode --classifier_name $classifier_name --class_type $class_type && echo 'SUCCESS' > '$status_file' || echo 'FAILURE' > '$status_file'"

            # Run the command directly in this session
            echo "Running $command"
            eval $command

            # Check if command was successful
            if [ "$(cat "$status_file")" == "SUCCESS" ]; then
                echo "Experiment $date completed successfully."
            else
                echo "Experiment $date failed."
                # Optionally, break the loop if you want to stop on failure
                # break 3
            fi

            # Stop after reaching maximum experiments
            if [ "$num" -gt "$max_experiments" ]; then
                break 3 # Exit all loops
            fi
        done
    done
done
