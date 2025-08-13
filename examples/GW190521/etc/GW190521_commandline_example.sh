#! /usr/bin/env bash

# Example for running tdinf in the commandline. NOT recommended due to computational expense, and NOT compatible with the `group_postprocess` or
# 'waveform_h5s' modules which easily load and plot results. 

# See GW190521.ini for documentation about arguments to `run_sampler.py` 

inputFolder=../GW190521_input_data
echo "Using inputFolder = ${inputFolder}"

mkdir output

runSamp=$(which run_sampler)

$runSamp \
    --output-h5 output/full_0.0seconds.h5 \
    --mode full \
    --Tcut-seconds 0.0 \
    --ifos H1 L1 V1 \
    --data H1:${inputFolder}/H-H1_GWOSC_16KHZ_R2-1242442952-32.hdf5 \
    --data L1:${inputFolder}/L-L1_GWOSC_16KHZ_R2-1242442952-32.hdf5 \
    --data V1:${inputFolder}/V-V1_GWOSC_16KHZ_R2-1242442952-32.hdf5 \
    --psd H1:${inputFolder}/glitch_median_PSD_H1.dat \
    --psd L1:${inputFolder}/glitch_median_PSD_L1.dat \
    --psd V1:${inputFolder}/glitch_median_PSD_V1.dat \
    --reference-posterior-file ${inputFolder}/GW190521_posterior_samples.h5 \
    --reference-parameter-method maxL \
    --total-mass-prior-bounds 200 400 \
    --mass-ratio-prior-bounds 0.17 1 \
    --spin-magnitude-prior-bounds 0 0.99 \
    --luminosity-distance-prior-bounds 100 10000 \
    --time-prior-sigma 0.01 \
    --approx NRSur7dq4 \
    --ncpu 128 \
    --nwalkers 512 \
    --nsteps 50000 \
    --Tstart 1242442966.907715 \
    --Tend 1242442967.607715 \
    --sampling-rate 2048 \
    --f22-start 0 \
    --fref 11 \
    --flow 11 \
    --fmax 1024 \
    --resume  \
    --vary-skypos  \
    --vary-time
