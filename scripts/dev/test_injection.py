#!/usr/bin/env python

from pylab import *
import argparse
import h5py
import lal
import scipy.linalg as sl
from collections import OrderedDict
from multiprocessing import Pool
from contextlib import closing
import os

import sys
sys.path.append('..')
import utils
from utils import reconstructwf as rwf
from utils import likelihood as ll


data_dir = '/home/simona.miller/time-domain-gw-inference/data/'

p = argparse.ArgumentParser()

p.add_argument('-t', '--Tcut-cycles', type=float, default=0) # defaults to the 0 as calculated from peak emission
p.add_argument('--Tstart', type=float, default=1242442966.9077148) 
p.add_argument('--Tend', type=float, default=1242442967.607715)
p.add_argument('--approx', default='NRSur7dq4')
p.add_argument('--downsample', type=int, default=8)
p.add_argument('--flow', type=float, default=11)
p.add_argument('--fref', type=float, default=11)
p.add_argument('--ifos', nargs='+', default=['H1', 'L1', 'V1'])
p.add_argument('--data-path', default=data_dir+'input/GW190521_data/{}-{}_GWOSC_16KHZ_R2-1242442952-32.hdf5')
p.add_argument('--psd-path', default=data_dir+'input/GW190521_data/glitch_median_PSD_{}.dat')
p.add_argument('--injected-parameters', default=data_dir+'input/injection_params/GW190521_maxP.json') 

args = p.parse_args()

'''
Script to test doing an injection
'''

ifos = args.ifos
psd_path = args.psd_path
      
# Load in injected parameters 
injected_parameters = utils.parse_injected_parameters(args.injected_parameters)

# Triggertime and sky position 
tpeak_geocent = injected_parameters['geocent_time']
skypos = {k:injected_parameters[k] for k in ['ra', 'dec', 'psi']}

## tpeak = placement of waveform
print('\nWaveform placement time:')
tpeak_dict, ap_dict = utils.get_tgps_and_ap_dicts(tpeak_geocent, ifos, skypos['ra'] , skypos['dec'], skypos['psi'])  

# PSDs 
pe_psds = {}
for ifo in ifos: 
    pe_psds[ifo] = genfromtxt(psd_path.format(ifo), dtype=float)

# Times
raw_time_dict = utils.load_raw_data(ifos=ifos,path=args.data_path)[0]

# Injection
raw_data_dict = utils.injectWaveform(parameters=injected_parameters, time_dict=raw_time_dict, 
                                           tpeak_dict=tpeak_dict, ap_dict=ap_dict, skypos=skypos, f_ref=args.fref)
              
## tcut = cutoff time in waveform
Ncycles = args.Tcut_cycles # find truncation time in # number of cycles from peak
# if 0, cut = peak
if Ncycles==0:
    tcut_geocent = tpeak_geocent
else:  
    tcut_geocent = utils.get_Tcut_from_Ncycles(Ncycles, parameters=injected_parameters, time_dict=raw_time_dict, 
                                               tpeak_dict=tpeak_dict, ap_dict=ap_dict, skypos=skypos, f_ref=args.fref)
    
print('\nCutoff time:')
tcut_dict, _ = utils.get_tgps_and_ap_dicts(tcut_geocent, ifos, skypos['ra'] , skypos['dec'], skypos['psi'])
    
# condition the data
ds_factor = args.downsample
f_low = args.flow
time_dict, data_dict, icut_dict = utils.condition(  # icut = index corresponding to cutoff time
    raw_time_dict, raw_data_dict, tcut_dict, ds_factor, f_low
)

# Time spacing of data 
dt = time_dict['H1'][1] - time_dict['H1'][0]

# Crop data
TPre = tcut_geocent - args.Tstart
TPost = args.Tend - tcut_geocent
    
Npre = int(round(TPre / dt))
Npost = int(round(TPost / dt)) + 1  # must add one so that the target time is actually included, even if Tpost = 0,
                                    # otherwise WF placement gets messed up   
Nanalyze = Npre + Npost
Tanalyze = Nanalyze*dt
print('\nWill analyze {:.3f} s of data at {:.1f} Hz\n'.format(Tanalyze, 1/dt))

# Crop analysis data to specified duration.
for ifo, idx in icut_dict.items():
    # idx = sample closest to desired time
    time_dict[ifo] = time_dict[ifo][idx-Npre:idx+Npost]
    data_dict[ifo] = data_dict[ifo][idx-Npre:idx+Npost]

# Make plot 

plt.figure(figsize=(20, 4))

for i, ifo in enumerate(ifos): 
    
    strain = data_dict[ifo]
    time = time_dict[ifo]
    
    plt.subplot(131+i)
    
    # plot strain
    plt.plot(time, strain, color=f'C{i}')
    
    # plot trigger time
    plt.axvline(tpeak_dict[ifo], color='k', ls='--', label='$t=0M$')
    
    # and cutoff time
    plt.axvline(tcut_dict[ifo], color='crimson', ls='--', label=f'$t=-{Ncycles}\,$cycles')
    
    plt.ylim(-6e-22, 6e-22)
    plt.ylabel('$h$')
    plt.xlabel('$t$')
    plt.title(ifo)
    
    if i==0: 
        plt.legend(handlelength=3)
        
plt.savefig(f'test_injection_GW190521_maxP_fref={args.fref}.pdf', bbox_inches='tight')