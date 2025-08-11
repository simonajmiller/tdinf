#! /usr/bin/env bash

wfs=$(which waveform_h5s)

# Optional: check if it's found
if [[ -z "$wfs" ]]; then
    echo "Error: waveform_h5s not found in PATH."
    exit 1
fi

uv run $wfs --directory output --run_key full --N_waveforms 1000 --ncpu 1

for t in -0.05 0 0.05: 
    uv run $wfs --directory output --run_key pre_${t}seconds --N_waveforms 1000 --ncpu 1
    uv run $wfs --directory output --run_key post_${t}seconds --N_waveforms 1000 --ncpu 1