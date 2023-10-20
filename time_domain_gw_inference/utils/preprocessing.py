import lal
import numpy as np
import h5py
from . import reconstructwf as rwf
from .spins_and_masses import *
import scipy.signal as sig
import json
import os
from collections import OrderedDict

def condition(raw_time_dict, raw_data_dict, t_dict, ds_factor=16, f_low=11,
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
    ds_factor : float (optional)
        downsampling factor for the data; defaults to 16 which takes ~16kHz data to 
        1024 Hz data
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
    
    # Cycle through interferometers
    for ifo in ifos:
        
        # Find the nearest sample in H to the designated time t
        i = np.argmin(np.abs(raw_time_dict[ifo] - t_dict[ifo]))
        ir = i % ds_factor
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
        if ds_factor > 1:
            if scipy_decimate:
                data = sig.decimate(data, ds_factor, zero_phase=True)
            else:
                data = data[::ds_factor]
            time = raw_time[::ds_factor]
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


def injectWaveform(**kwargs): 
    
    # Unpack inputs
    p = kwargs.pop('parameters')
    time_dict = kwargs.pop('time_dict')
    tpeak_dict = kwargs.pop('tpeak_dict')
    ap_dict = kwargs.pop('ap_dict')
    skypos = kwargs.pop('skypos')
    approx = kwargs.pop('approx', 'NRSur7dq4')
    f_low = kwargs.pop('f_low')
    f_ref = kwargs.pop('f_ref')
    ifos = kwargs.pop('ifos', ['H1', 'L1', 'V1'])
    
    # Get dt 
    dt = time_dict['H1'][1] - time_dict['H1'][0]
    
    # Change spin convention
    iota, s1x, s1y, s1z, s2x, s2y, s2z = transform_spins(p['theta_jn'], p['phi_jl'], p['tilt_1'], p['tilt_2'],
                                          p['phi_12'], p['a_1'], p['a_2'], p['mass_1'], p['mass_2'],
                                          f_ref, p['phase'])

    # Get strain
    hp, hc = rwf.generate_lal_hphc(approx, 
                                   p['mass_1'], p['mass_2'],
                                   [s1x, s1y, s1z], 
                                   [s2x, s2y, s2z],
                                   dist_mpc=p['luminosity_distance'], 
                                   dt=dt,
                                   f_low=f_low, 
                                   f_ref=f_ref,
                                   inclination=iota,
                                   phi_ref=p['phase']
                                  )
    
    # Project into each detector 
    h_ifos = {}
    for ifo in ifos:

        # Time align 
        h = rwf.generate_lal_waveform(hplus=hp, hcross=hc, times=time_dict[ifo], triggertime=tpeak_dict[ifo])

        # Project onto H1
        Fp, Fc = ap_dict[ifo]
        h_ifo = Fp*h.real - Fc*h.imag
        
        h_ifos[ifo] = h_ifo
        
    return h_ifos


def get_Tcut_from_Ncycles(Ncycles, **kwargs): 
    
    """
    Calculate the cutoff time given the cutoff cycle and the parameters of the 
    waveform to base the cutoff time from
    """
    
    # Get waveform in H1
    h_H1 = injectWaveform(**kwargs)['H1']
    
    # Get indices of extrema 
    idxs, _ = sig.find_peaks(np.abs(h_H1), height=0)
    
    # Get times of extrema 
    times = kwargs['time_dict']['H1']
    t_cycles_H1 = times[idxs]
    
    # Get the cycle we care about
    i0 = np.argmax(np.abs(h_H1[idxs])) # index corresponding to merger (absolute peak time)
    n_i = 2*Ncycles                    # one index = 1/2 cycle
    assert(n_i.is_integer()), '# of half cycles does not correspond to an integer value'
    icut = i0 + int(n_i)               # index corresponding to the cycle we care about
    
    # Get time in H1
    tcut_H1 = t_cycles_H1[icut]
    
    # Get geocenter time
    skypos = kwargs['skypos']
    dt_H = lal.TimeDelayFromEarthCenter(lal.cached_detector_by_prefix['H1'].location, 
                                        skypos['ra'], skypos['dec'], lal.LIGOTimeGPS(tcut_H1))
    tcut_geo = tcut_H1-dt_H
    
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

        # lower freq cut
        m = freq >= f_low
        psd[~m] = 100 * max(psd[m])  # set values below 11 Hz to be equal to 100*max(psd)

        # upper freq cut
        fmax = 0.5 / dt        
        freq = freq[freq <= fmax]
        psd = psd[freq <= fmax]

        # Computer ACF from PSD
        rho = 0.5 * np.fft.irfft(psd) / dt  # dt comes from numpy fft conventions
        rho_dict[ifo] = rho[:Nanalyze]
        
    return rho_dict