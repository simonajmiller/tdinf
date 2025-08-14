#! /usr/bin/env bash

# Example for running TDinf code using the condor pipeline

tdinf_condor_pipe \
    --modes full pre post \
    --times_list -0.05 0 0.05 \
    --output_directory output \
    --config_file GW190521.ini \
    --run_in_place \
    --submit

# -------------------------------
# Explanation of arguments 
# --------------------------------
# --modes full pre post      | This runs on full signal, and pre/post t = {-0.05, 0, 0.05} seconds from merger. 
# --times_list -0.05 0 0.05  | NOTE: Could alternatively pass --cycles_list to cut in # of cycles rather than seconds.
# --output_directory output  | The output is put in a folder called `output` inside the current directory. 
# --config_file GW190521.ini | Path to the config file.
# --run_in_place | Skip condor file transfer; use if on a shared file system.
# --submit       | Submit job to condor automatically. If excluded, run manually with: ./output/submit.sh
