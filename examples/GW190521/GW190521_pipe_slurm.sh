#! /usr/bin/env bash

# Example for running TDinf code using the slurm pipeline. 

# -------------------------------
# Explanation of arguments 
# --------------------------------
# --modes full pre post      | This runs on full signal, and pre/post t = {-0.05, 0, 0.05} seconds from merger. 
# --times_list -0.05 0 0.05  | NOTE: Could alternatively pass --cycles_list to cut in # of cycles rather than seconds.
# --output_directory output  | The output is put in a folder called `output` inside the current directory. 
# --config_file GW190521.ini | Path to the config file.
# --overwrite   | If `output` directory already exists, overwrite it.
# --time 3-0    | Request three days for running.
# --ntasks 7    | Full + pre three times + post three times = 7 tasks for slurm to run.
# --submit      | Submit job to slurm automatically. If excluded, then must manually run: ./output/submit.sh

uv run time_domain_gw_inference_slurm_pipe \
    --modes full pre post \
    --times_list -0.05 0 0.05 \
    --output_directory output \
    --config_file GW190521.ini \
    --overwrite \
    --time 3-0 \
    --ntasks 7 \
    --submit