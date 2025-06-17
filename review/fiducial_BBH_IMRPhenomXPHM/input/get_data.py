import numpy as np
import json 
from gwpy.timeseries import TimeSeries
import h5ify
import os

# copy the metadata from CIT
computer = 'simona.miller@ldas-pcdev2.ligo.caltech.edu'
meta_fname = '/home/pe.o4/sampler-review/data/fiducial/XPHM-BBH/meta.json'
os.system(f"scp {computer}:{meta_fname} .")  

# interferometers
ifos = ['H1', 'L1', 'V1']

# load in the metadata
with open('meta.json') as f:
    metadata = json.load(f)

# get and save the injected parameters
with open('injection.json', 'w') as f:
    json.dump(metadata['injection'], f)

# get the PSDs
ASD_filenames = {ifo:metadata[ifo]['asd_file'] for ifo in ifos}
for ifo, fname in ASD_filenames.items(): 

    # copy from CIT
    os.system(f"scp {computer}:{fname} .")  
    
    # load ASD
    freq_ASD = np.loadtxt(os.path.basename(fname))
    
    # ASD --> PSD
    freq_PSD = freq_ASD.copy()
    freq_PSD[:,1] = freq_ASD[:,1]**2
    
    # save 
    np.savetxt(f'{ifo}_PSD.txt', freq_PSD)

# get the strain data
strain_filesnames = {ifo:metadata[ifo]['filename'] for ifo in ifos}
channels = {ifo:metadata[ifo]['channel'] for ifo in ifos}
for ifo in ifos:  

    # copy from CIT
    fname = strain_filesnames[ifo]
    os.system(f"scp {computer}:{fname} .") 
    
    # load in strain data from gwf
    gwf_data = TimeSeries.read(os.path.basename(fname), channels[ifo])
    
    # convert to a dictionary
    t0_gen = metadata['injection']['geocent_time']
    mask = (gwf_data.times.value > t0_gen - 2) & (gwf_data.times.value <  t0_gen + 2)
    data = {'strain':gwf_data.value[mask], 'times':gwf_data.times.value[mask]}

    # save as .h5 file
    h5ify.save(f'{ifo}_strain.h5', data)