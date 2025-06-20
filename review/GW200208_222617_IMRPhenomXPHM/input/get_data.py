import numpy as np
import json 
from gwpy.timeseries import TimeSeries
import h5ify
import os
import sys
from time_domain_gw_inference import utils

## compare to: 
## https://ldas-jobs.ligo.caltech.edu/~richard.george/bilby_review_013023/combo_results_v4/html/dynesty_dynesty_Config.html

# CIT
computer = 'simona.miller@ldas-pcdev2.ligo.caltech.edu'

# Which ifos?
ifos = ['H1', 'L1', 'V1']

# Copy PSDs
for ifo in ifos: 
    if not os.path.exists(f'{ifo}-psd.dat'):
        os.system(f"scp {computer}:/home/aaron.zimmerman/review/BilbyO4/GW200208_22/psds/{ifo}-psd.dat .")

# Get reference posterior from Zenodo
ref_posterior_fname = "IGWN-GWTC3p0-v1-GW200208_222617_PEDataRelease_mixed_cosmo.h5"
if not os.path.exists(ref_posterior_fname):
    os.system(f"wget https://zenodo.org/record/5546663/files/{ref_posterior_fname}")

# Load reference posterior and get t0 
ref_posterior = h5ify.load(ref_posterior_fname)['C01:IMRPhenomXPHM']['posterior_samples']
ref_parameters = utils.get_reference_parameters_from_posterior(ref_posterior)
t0 = ref_parameters['geocent_time']

# print some info for determining priors
print(f't0 = {t0}')
print('total mass:', min(ref_posterior['total_mass']), max(ref_posterior['total_mass']))
print('luminosity distance:', min(ref_posterior['luminosity_distance']), max(ref_posterior['luminosity_distance']))


# Get strain data
strain_filesnames = {
    'H1':'/home/aaron.zimmerman/review/BilbyO4/GW200208_22/frames/H-H1_GWOSC_16KHZ_R1-1265233948-4096.gwf',
    'L1':'/home/aaron.zimmerman/review/BilbyO4/GW200208_22/frames/L-L1_GWOSC_16KHZ_R1-1265233948-4096.gwf',
    'V1':'/home/aaron.zimmerman/review/BilbyO4/GW200208_22/frames/V-V1_GWOSC_16KHZ_R1-1265233948-4096.gwf'
}
channels = {
    'H1':'H1:GWOSC-16KHZ_R1_STRAIN',
    'L1':'L1:GWOSC-16KHZ_R1_STRAIN',
    'V1':'V1:GWOSC-16KHZ_R1_STRAIN'
}
for ifo in ifos:  

    fname = strain_filesnames[ifo]

    # copy from CIT
    if not os.path.exists(os.path.basename(fname)):
        os.system(f"scp {computer}:{fname} .") 

    # format correctly for TD code
    if not os.path.exists(f'{ifo}_strain.h5'):

        # load in strain data from gwf
        gwf_data = TimeSeries.read(os.path.basename(fname), channels[ifo])
        
        # convert to a dictionary
        mask = (gwf_data.times.value > t0 - 2) & (gwf_data.times.value <  t0 + 2)
        data = {'strain':gwf_data.value[mask], 'times':gwf_data.times.value[mask]}
    
        # save as .h5 file
        h5ify.save(f'{ifo}_strain.h5', data)