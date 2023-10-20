#! /usr/bin/env bash

# get strain
for IFO in H L V; do
    wget https://www.gw-openscience.org/eventapi/html/O3_Discovery_Papers/GW190521/v2/${IFO}-${IFO}1_GWOSC_16KHZ_R2-1242442952-32.hdf5
done

# get PE samples
wget https://dcc.ligo.org/public/0168/P2000158/004/GW190521_posterior_samples.h5
