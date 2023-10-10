#!/usr/bin/env python

from pylab import *
import argparse
import h5py
import lal
import emcee
import scipy.linalg as sl
from collections import OrderedDict
import pandas as pd
from multiprocessing import Pool
from contextlib import closing
import os
from tqdm import tqdm
import utils
from utils import reconstructwf as rwf
from utils import likelihood as ll
import sys

data_dir = '/home/simona.miller/time-domain-gw-inference/data/'

"""
Parse agruments
"""

p = argparse.ArgumentParser()

# Required args: where to save data ...
p.add_argument('-o', '--output')
# ... and whether to run pre-Tcut, post-Tcut, or full (Tstart to Tend)?
p.add_argument('-m', '--mode')

# Args for cutoff (defined in # of cycles), start, & end times
p.add_argument('-t', '--Tcut-cycles', type=float, default=0) # defaults to the 0 as calculated from peak emission
p.add_argument('--Tstart', type=float, default=1242442966.9077148) 
p.add_argument('--Tend', type=float, default=1242442967.607715)

# Optional args to specify waveform, data, and sampler settings
p.add_argument('--nwalkers', type=int, default=200)
p.add_argument('--nsteps', type=int, default=1000)
p.add_argument('--ncpu', type=int, default=4)
p.add_argument('--approx', default='NRSur7dq4')
p.add_argument('--downsample', type=int, default=8)
p.add_argument('--flow', type=float, default=11)
p.add_argument('--fref', type=float, default=11)
p.add_argument('--ifos', nargs='+', default=['H1', 'L1', 'V1'])
p.add_argument('--data-path', default=data_dir+'input/GW190521_data/{}-{}_GWOSC_16KHZ_R2-1242442952-32.hdf5')
p.add_argument('--psd-path', default=data_dir+'input/GW190521_data/glitch_median_PSD_{}.dat') ## for LI 

# Option to do an injection instead of use real data;
# if "REALDATA", do not do an injection, else file path to injected parameters
p.add_argument('--injected-parameters', default="REALDATA")

# Do we want to run with only the prior?
p.add_argument('--only-prior', action='store_true')

# Do we want to sample in time and/or sky position?
p.add_argument('--vary-time', action='store_true')
p.add_argument('--vary-skypos', action='store_true')

# Do we want to resume an old run? 
p.add_argument('--resume', action='store_true')

args = p.parse_args()

# Check that the given mode is allowed 
run_mode = args.mode 
assert run_mode in ['full', 'pre', 'post'], f"mode must be 'full', 'pre', or 'post'. given mode = '{run_mode}'."

# Unpack some basic parameters
ifos = args.ifos
psd_path = args.psd_path
f_ref = args.fref
f_low = args.flow
ds_factor = args.downsample

print('') # for printing aesthetics lol
    
"""
Load or generate data
"""

# If real data ...
if args.injected_parameters == "REALDATA": 
    
    # Load data
    raw_time_dict, raw_data_dict = utils.load_raw_data(ifos=ifos,path=args.data_path)
    pe_out = utils.get_pe(raw_time_dict, verbose=False, psd_path=psd_path)
    tpeak_geocent, pe_samples, log_prob, pe_psds, skypos = pe_out
    
    # "Injected parameters" = max(P) draw from the samples associated with this data
    injected_parameters = pe_samples[np.argmax(log_prob)]
    
    ## tpeak = placement of waveform
    print('\nWaveform placement time:')
    tpeak_dict, ap_dict = utils.get_tgps_and_ap_dicts(tpeak_geocent, ifos, skypos['ra'] , skypos['dec'], skypos['psi'])  

# Else, generate an injection (currently, only set up for no noise case)
else:    
    
    # Load in injected parameters 
    injected_parameters = utils.parse_injected_parameters(args.injected_parameters)
    
    # Check that the reference freqs line up 
    err_msg = f"Injection fref={injected_parameters['f_ref']} does not equal sampler fref={f_ref}"
    assert injected_parameters['f_ref'] == f_ref, err_msg
    
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
                                         tpeak_dict=tpeak_dict, ap_dict=ap_dict, skypos=skypos, 
                                         f_ref=f_ref, f_low=f_low)
              
## tcut = cutoff time in waveform
Ncycles = args.Tcut_cycles # find truncation time in # number of cycles from peak
# if 0, cut = peak
if Ncycles==0:
    tcut_geocent = tpeak_geocent
else:  
    tcut_geocent = utils.get_Tcut_from_Ncycles(Ncycles, parameters=injected_parameters, time_dict=raw_time_dict, 
                                               tpeak_dict=tpeak_dict, ap_dict=ap_dict, skypos=skypos, f_ref=f_ref, f_low=f_low)
    
print('\nCutoff time:')
tcut_dict, _ = utils.get_tgps_and_ap_dicts(tcut_geocent, ifos, skypos['ra'] , skypos['dec'], skypos['psi'])
    
# If we are varying skyposition 
if args.vary_skypos: 
    ap_dict = None # don't want fixed antenna patterns
    
# If we are varying over time of coalescence 
if args.vary_time:
    tpeak_dict = None # don't want fixed time of arrival at detectors
    
    
"""
Condition data
"""

# icut = index corresponding to cutoff time
time_dict, data_dict, icut_dict = utils.condition(raw_time_dict, raw_data_dict, tcut_dict, ds_factor, f_low) 

# Time spacing of data 
dt = time_dict['H1'][1] - time_dict['H1'][0]

# Decide how much data to analyze based of off run mode 
if run_mode == 'full': 
    TPre = tcut_geocent - args.Tstart
    TPost = args.Tend - tcut_geocent
elif run_mode == 'pre': 
    TPre = tcut_geocent - args.Tstart
    TPost = 0
elif run_mode == 'post': 
    TPre = 0
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

    
"""
Calculate ACF
"""

# Condition PSDs
cond_psds = {}
for ifo, freq_psd in pe_psds.items():
    freq, psd = freq_psd.copy().T
    
    # lower freq cut 
    m = freq >= 11
    psd[~m] = 100*max(psd[m]) # set values below 11 Hz to be equal to 100*max(psd)    
    
    # upper freq cut 
    fmax = 0.5 / dt 
    m2 = freq <= fmax
    
    cond_psds[ifo] = (freq[m2], psd[m2])


rho_dict = OrderedDict() # stores acf 
L_dict = OrderedDict()   # stores L such that cov matrix C = L^T L

for ifo in ifos:
    
    # Get PSD
    freq, psd = cond_psds[ifo]
    
    # Make sure the max freq of PSD and dt correspond correctly 
    err_msg = f'time spacing ({dt}) not equal to 0.5 / max(freq) ({0.5 / round(freq.max())})'
    assert dt == 0.5 / round(freq.max()), err_msg
    
    # Computer ACF from PSD
    rho = 0.5*np.fft.irfft(psd) / dt # dt comes from numpy fft conventions
    rho_dict[ifo] = rho[:Nanalyze]
    
    # compute covariance matrix  C and its Cholesky decomposition L (~sqrt of C)
    C = sl.toeplitz(rho[:Nanalyze])
    L_dict[ifo] = np.linalg.cholesky(C)
    
"""
Arguments for the posterior function
"""

kwargs = {
    'mtot_lim' : [200, 350],
    'q_lim' : [0.17, 1],
    'chi_lim' : [0, 0.99],
    'dist_lim' : [1000, 10000],
    'approx' : args.approx,
    'f_ref' : f_ref,
    'f_low' : f_low,
    'only_prior' : args.only_prior,
    'delta_t' : dt,
    'ra' : skypos['ra'],   # default right ascension if not varied
    'dec' : skypos['dec'], # default declination if not varied
    'psi' : skypos['psi'], # default polarization if not varied
    'tgps_geocent' : tpeak_geocent # default waveform placement time if not varied
}
kwargs.update({k: globals()[k] for k in ['rho_dict', 'time_dict', 'data_dict',
                                         'ap_dict', 'tpeak_dict']})

        
"""
Set up sampler
"""

# Emcee args 
nsteps = args.nsteps
nwalkers = args.nwalkers
# Default num dimensions (fixed time and sky position) = 14
ndim = 14
# If we want to vary time ...
if args.vary_time:
    ndim += 1
# If we want to vary sky position ...
if args.vary_skypos:
    ndim += 5 # add ra_x, ra_y, sin_dec, psi_x, psi_y

print("Sampling %i parameters." % ndim)

# Where to save samples while sampler running
backend_path = data_dir + 'output/' + args.output
backend = emcee.backends.HDFBackend(backend_path)

# Resume if we want 
if args.resume and os.path.isfile(backend_path):
    # Load in last sample to use as the new starting walkers
    p0 = backend.get_last_sample()
else:
    # Reset the backend
    backend.reset(nwalkers, ndim)

    # Initialize walkers 
    # (code sees unit scale quantities; use logit transformations
    # to take boundaries to +/- infinity)
    p0_arr = np.asarray([[np.random.normal() for j in range(ndim)] for i in range(nwalkers)])

    # if time of coalescence sampled over need to include this separately since it isn't a unit scaled quantity
    if args.vary_time:
        dt_1M = 0.00127
        sigma_time = dt_1M*2.5 # time prior from LVK has width of ~2.5M
        initial_t_walkers = np.random.normal(loc=tpeak_geocent, scale=sigma_time, size=nwalkers) 
        p0_arr[:,ndim-1] = initial_t_walkers  # time always saved as the final param 

    p0 = p0_arr.tolist()

# Deactivate numpy default number of cores to avoid using too many 
if args.ncpu > 1:
    os.environ["OMP_NUM_THREADS"] = "1"

# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(nsteps)

# This will be useful for testing convergence
old_tau = np.inf

# now use multiprocessing
def sort_on_runtime(pos):
    p = np.atleast_2d(pos)
    idx = np.argsort(p[:, 0])[::-1]
    return p[idx], idx
    
    
"""
Run sampler
"""

# Now we'll sample for up to max_n steps
print("Running with %i cores." % args.ncpu)
with closing(Pool(processes=args.ncpu)) as pool:
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ll.get_lnprob,
                                    backend=backend, pool=pool,
                                    runtime_sortingfn=sort_on_runtime,
                                    kwargs=kwargs)

    for sample in sampler.sample(p0, iterations=nsteps, progress=True):
       
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break # if convergence reached before nsteps, stop running
        old_tau = tau
        
    pool.terminate()
    
    
"""
Post processing and saving data
"""
# Print dimensions of chain
print(sampler.get_chain().shape)
    
# Post processing
tau = sampler.get_autocorr_time(quiet=True)
burnin = int(5*np.max(tau))
thin = int(0.5*np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
samples_dict = ll.get_dict_from_samples(samples, angles=True, **kwargs)

# Posteriors
samples_lnp = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
samples_dict['ln_posterior'] = samples_lnp
samples_lnprior = np.asarray([ll.get_lnprior(x, **kwargs) for x in samples])
samples_dict['ln_prior'] = samples_lnprior

# Turn into data frame
df = pd.DataFrame(samples_dict)
df = df[[k for k,v in df.items() if v.min() != v.max()]]

# Add info about tpeak, tcut, and injected parameters
if not args.vary_time:
    tpeak_dict['geocenter'] = tpeak_geocent
    df.attrs['t_peak'] = tpeak_dict
tcut_dict['geocenter'] = tcut_geocent
df.attrs['t_cut'] = tcut_dict
df.attrs['skypos'] = skypos
df.attrs['injected_parameters'] = injected_parameters

# Save
dat_path = backend_path.replace('h5', 'dat')
df.to_csv(dat_path, sep=' ', index=False)
print("File saved: %r" % dat_path)