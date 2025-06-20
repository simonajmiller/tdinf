import numpy as np
import json 
from gwpy.timeseries import TimeSeries
import h5ify
import os
import sys
from time_domain_gw_inference import utils

# CIT
computer = 'simona.miller@ldas-pcdev2.ligo.caltech.edu'

# Get reference posterior from Zenodo
ref_posterior_fname = "IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
if not os.path.exists(ref_posterior_fname):
    os.system(f"wget https://zenodo.org/record/6513631/files/{ref_posterior_fname}")

# Copy PSDs
if not os.path.exists('H1-psd.dat'):
    os.system(f"scp {computer}:/home/daniel.williams/events/O3/event_repos/GW150914/C01_offline/psds/2048/H1-psd.dat .")
if not os.path.exists('L1-psd.dat'):
    os.system(f"scp {computer}:/home/daniel.williams/events/O3/event_repos/GW150914/C01_offline/psds/2048/L1-psd.dat .")

# Get strain data
# See /home/daniel.williams/events/O3/o3a/run_directories/GW150914/ProdF4/log_data_generation/ProdF4_data0_1126259462-391_generation.err on CIT
channels = {'H1':'H1:DCS-CALIB_STRAIN_C02', 'L1':'L1:DCS-CALIB_STRAIN_C02'}
for ifo, channel in channels.items(): 

    if not os.path.exists(f'{ifo}_strain.h5'):

        try:
            # load in strain data
            gwf_data = TimeSeries.get(channel, start=1126259460.391, end=1126259464.391, dtype='float64')
            
            # convert to a dictionary
            data = {'strain':gwf_data.value, 'times':gwf_data.times.value}
        
            # save as .h5 file
            h5ify.save(f'{ifo}_strain.h5', data)

        except: 
            os.system(f"scp {computer}:/home/simona.miller/misc/GW150914_data/{ifo}_strain.h5 .")