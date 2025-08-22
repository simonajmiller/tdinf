#! /usr/bin/env bash

wfs=$(which waveform_h5s)

# Check if it's found
if [[ -z "$wfs" ]]; then
    echo "Error: waveform_h5s not found in PATH."
    exit 1
fi

# Generate waveforms for full run
cd ../output
$wfs --directory . --run_key full --N_waveforms 1000 --ncpu 1

# Generate waveforms for pre/post cutoff runs 
for t in -0.05 0.0 0.05; do
    $wfs --directory . --run_key pre_${t} --N_waveforms 1000 --ncpu 1
    $wfs --directory . --run_key post_${t} --N_waveforms 1000 --ncpu 1
done

cd - 
