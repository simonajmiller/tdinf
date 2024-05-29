import lal
import numpy as np
import h5py
from . import reconstructwf as rwf
from .spins_and_masses import *
import scipy.signal as sig
import json
import os
from collections import OrderedDict

def condition(raw_time_dict, raw_data_dict, t_dict, desired_sample_rate, f_low=11,
              scipy_decimate=True, verbose=True):
    
    """
    Filter and downsample the data, and locate target sample corresponding
    to the times in t_dict
    
    Parameters
    ----------
    raw_time_dict : dictionary
        time stamps for the raw strain data from each ifo (output from load_raw_data 
        function above)
    raw_data_dict : dictionary
        the raw strain data data from each ifo (output from load_raw_data function 
        above)
    t_dict : dictionary
        time at each interferometer find the sample index of
    desired_sample_rate : float must be a factor of 2^n
        sampling rate that we want the data to have after we downsample the original data
    f_low : float (optional)
        frequency for the highpass filter
    scipy_decimate : boolean (optional)
        whether or not to use the scipy decimate function for downsampling, defaults
        to True 
    verbose : boolean (optional)
        whether or not to print out information calculated
        
    Returns
    -------
    time_dict : dictionary
        time stamps for the conditioned strain data from each ifo 
    data_dict : dictionary
        the conditioned strain data from each ifo 
    i_dict : dictionary
        indices corresponding to the time values in t_dict
    """
    
    ifos = list(raw_time_dict.keys())
    data_dict = {}
    time_dict = {}
    i_dict = {}

    ifo = list(raw_time_dict.keys())[0]
    raw_data_sample_rate = 1 / (raw_time_dict[ifo][1] - raw_time_dict[ifo][0])
    # TODO how do i check that this is an integer?
    downsample_factor = int(np.floor(raw_data_sample_rate / desired_sample_rate))
    if verbose:
        print('downsample factor is', downsample_factor)
    if downsample_factor == 0:
        raise ValueError(f"Desired sampling rate must be less than or equal to given sample rate! "
                         f"raw:{raw_data_sample_rate}, desired:{desired_sample_rate}")

    
    # Cycle through interferometers
    for ifo in ifos:
        
        # Find the nearest sample in H to the designated time t
        i = np.argmin(np.abs(raw_time_dict[ifo] - t_dict[ifo]))
        ir = i % downsample_factor
        if verbose:
            print('\nRolling {:s} by {:d} samples'.format(ifo, ir))
        raw_data = np.roll(raw_data_dict[ifo], -ir)
        raw_time = np.roll(raw_time_dict[ifo], -ir)
        
        # Filter
        if f_low:
            fny = 0.5/(raw_time[1]-raw_time[0])
            b, a = sig.butter(4, f_low/fny, btype='highpass', output='ba')
            data = sig.filtfilt(b, a, raw_data)
        else:
            data = raw_data.copy()
        
        # Decimate
        if downsample_factor > 1:
            if scipy_decimate:
                data = sig.decimate(data, downsample_factor, zero_phase=True)
            else:
                data = data[::downsample_factor]
            time = raw_time[::downsample_factor]
        else: 
            time = raw_time
        
        # Subtract mean and store
        data_dict[ifo] = data - np.mean(data)
        time_dict[ifo] = time
        
        # Locate target sample
        i_dict[ifo] = np.argmin(np.abs(time - t_dict[ifo]))
        if verbose:
            print('tgps_{:s} = {:.6f}'.format(ifo, t_dict[ifo]))
            print('t_{:s} - tgps_{:s} is {:.2e} s'.format(ifo, ifo, time[i_dict[ifo]]-t_dict[ifo]))
            
    return time_dict, data_dict, i_dict


def injectWaveform(injection_approx, **kwargs):
    # TODO modify this function so that it directly invokes the waveformManager we wrote before
    # Also, this currently does not depend on skypos and i think it maybe should? something to look into
    # Unpack inputs
    p = kwargs.pop('parameters')
    time_dict = kwargs.pop('time_dict')
    tpeak_dict = kwargs.pop('tpeak_dict')
    ap_dict = kwargs.pop('ap_dict')
    skypos = kwargs.pop('skypos')
    f22_start = kwargs.pop('f22_start')
    f_ref = kwargs.pop('f_ref')
    ifos = kwargs.pop('ifos', ['H1', 'L1', 'V1'])
    
    # Get dt 
    dt = time_dict['H1'][1] - time_dict['H1'][0]
    
    # Change spin convention
    iota, s1x, s1y, s1z, s2x, s2y, s2z = transform_spins(p['theta_jn'], p['phi_jl'], p['tilt_1'], p['tilt_2'],
                                          p['phi_12'], p['a_1'], p['a_2'], p['mass_1'], p['mass_2'],
                                          f_ref, p['phase'])
    if p['phi_jl'] == 0:
        s1x, s1y, s2x, s2y = 0, 0, 0, 0

    # Get strain
    hp, hc = rwf.generate_lal_hphc(injection_approx,
                                   p['mass_1'], p['mass_2'],
                                   [s1x, s1y, s1z], 
                                   [s2x, s2y, s2z],
                                   dist_mpc=p['luminosity_distance'], 
                                   dt=dt,
                                   f22_start=f22_start,
                                   f_ref=f_ref,
                                   inclination=iota,
                                   phi_ref=p['phase']
                                  )
    
    # Project into each detector 
    h_ifos = {}
    for ifo in ifos:

        # Time align 
        h = rwf.generate_lal_waveform(hplus=hp, hcross=hc, times=time_dict[ifo], triggertime=tpeak_dict[ifo])

        # Project using antenna partterns
        Fp, Fc = ap_dict[ifo]
        h_ifo = Fp*h.real - Fc*h.imag
        
        h_ifos[ifo] = h_ifo
        
    return h_ifos


def get_Tcut_from_Ncycles(waveform_dict, time_dict, ifo, Ncycles, ra, dec):
    
    """
    Calculate the cutoff time given the cutoff cycle and the parameters of the 
    waveform to base the cutoff time from
    """

    # Get waveform in H1
    h_ifo = waveform_dict[ifo]
    
    # Get indices of extrema 
    idxs, _ = sig.find_peaks(np.abs(h_ifo), height=0)
    
    # Get times of extrema 
    times = time_dict[ifo]
    t_cycles_ifo = times[idxs]
    
    # Get the cycle we care about
    i0 = np.argmax(np.abs(h_ifo[idxs])) # index corresponding to tcut=0 (absolute peak time)
    n_i = 2 * Ncycles                    # one index = 1/2 cycle

    # If the desired cycle cut is at a peak/trough ...
    if n_i.is_integer():
        
        icut = i0 + int(n_i)           # index corresponding to the cycle we care about

        # Get time in H1
        tcut_ifo = t_cycles_ifo[icut]
    
    # Otherwise, linearly interpolate between nearest peak and trough  
    else: 
        # Our desired cut sits between these two times 
        tcut_ifo_min = t_cycles_ifo[i0 + int(np.floor(n_i))]
        tcut_ifo_max = t_cycles_ifo[i0 + int(np.ceil(n_i))]

        # How far between the extrema?
        frac_between = n_i - np.floor(n_i)

        # Interpolate
        tcut_ifo = tcut_ifo_min + frac_between*(tcut_ifo_max - tcut_ifo_min)
    
    # Get geocenter time
    dt_ifo = lal.TimeDelayFromEarthCenter(lal.cached_detector_by_prefix[ifo].location,
                                        ra, dec, lal.LIGOTimeGPS(tcut_ifo))
    tcut_geo = tcut_ifo - dt_ifo
    
    return tcut_geo


def get_ACF(pe_psds, time_dict, f_low=11): 
    
    """
    Get ACF from PSDs and the times
    """
    
    rho_dict = OrderedDict()  # stores acf
    
    # Cycle through ifos
    for ifo, freq_psd in pe_psds.items():
                
        # unpack psd
        freq, psd = freq_psd.copy().T
        
        # Get dt
        times = time_dict[ifo]
        dt = times[1] - times[0]
        Nanalyze = len(times)

        # set values outside of upper and lower bound to 100xmaxPSD value
        fmax = 0.5 / dt   # upper freq cut (nyquist)
        m = (freq >= f_low) & (freq <= fmax)  # Combine conditions for both lower and upper frequency cuts
        psd[~m] = 100 * max(psd[m])  # set values outside the desired frequency range to be equal to 100*max(psd)

        freq = freq[freq <= fmax]
        psd = psd[freq <= fmax]
        # Compute ACF from PSD
        rho = 0.5 * np.fft.irfft(psd) / dt  # dt comes from numpy fft conventions
        rho_dict[ifo] = rho[:Nanalyze]
        
    return rho_dict
