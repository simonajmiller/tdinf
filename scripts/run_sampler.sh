#!/bin/bash

. /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
conda activate lalsuite-tm100
. /home/simona.miller/opt/lalsuite_tm100/etc/lalsuite-user-env.sh

savename=$1
runmode=$2
ncycles=$3
nproc=$4
nsteps=$5
nwalkers=$6
injection=$7

python /home/simona.miller/time-domain-gw-inference/scripts/run_sampler.py \
        -o $savename.h5 \
        -m $runmode \
        -t $ncycles \
        --ncpu $nproc \
        --nsteps $nsteps \
        --nwalkers $nwalkers \
        --injected-parameters $injection \
        --resume

conda deactivate