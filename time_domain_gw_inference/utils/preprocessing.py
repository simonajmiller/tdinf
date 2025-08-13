import lal
import numpy as np
import h5py
from .spins_and_masses import *
import scipy.signal as sig
import json
import os
from collections import OrderedDict
import re

def condition(raw_time_dict, raw_data_dict, t_dict, desired_sample_rate, f_min=11, f_max=None,
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
        frequency for the lower bound of the band pass / highpass
    f_max : float (optional)
        frequency for the upper bound of the band pass / lowpass
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

    # Get sampling rate
    ifo = list(raw_time_dict.keys())[0]
    raw_data_sample_rate = 1 / (raw_time_dict[ifo][1] - raw_time_dict[ifo][0])

    # Get downsample factor
    _downsample_factor = raw_data_sample_rate / desired_sample_rate
    assert _downsample_factor.isinteger(), f'Downsample factor must be a interger, but is instead {_downsample_factor}'
    downsample_factor = int(np.floor(_downsample_factor))
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

        # Nyquist frequency
        fny = 0.5/(raw_time[1] - raw_time[0])
        
        # Filter
        b, a = sig.butter(4, f_min/fny, btype='highpass', output='ba')
        data = sig.filtfilt(b, a, raw_data)
        
        # Decimate
        if downsample_factor > 1:
            if scipy_decimate:
                # sig.decimate includes a lowpass filter at target fny!
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


def get_reference_parameters_from_posterior(ref_pe_samples, method='tightest_time_posterior'): 
    """
    Find reference parameters from a set of samples (`ref_pe_samples`). This determines t0. 
    Options for `method` include 'tightest_time_posterior' or 'maxL'.
    """
    err_msg =  f'Unknown method {method} for calculating reference sample from reference posterior.'
    assert method in ['maxL', 'tightest_time_posterior'], err_msg

    # get the parameter names from the posterior
    parameter_names = ref_pe_samples.dtype.names

    if method=='maxL':
        '''
        Reference sample = maximum likelihood sample
        '''
        # Use this to get the reference sample
        ii = np.argmax(ref_pe_samples['log_likelihood'])

    elif method=='tightest_time_posterior': 
        '''
        Reference sample calculated as follows:
        1. Calculate standard deviation of the posterior for time in each detector
        2. Find which time posterior is the narrowest
        3. Take its median.
        4. Whichever sample has the closest time to that median is the reference sample.
        '''
        # find H1_time, L1_time, and/or V1_time
        pattern = re.compile(r'.*1_time$')
        ifo_time_posteriors = [s for s in parameter_names if pattern.match(s)]
        
        # get widths of time posteriors
        ifo_time_posterior_widths = [np.std(ref_pe_samples[ifo]) for ifo in ifo_time_posteriors]
    
        # find the narrowest one
        narrowest = ifo_time_posteriors[np.argmin(ifo_time_posterior_widths)]
        narrowest_posterior = ref_pe_samples[narrowest]
        
        print(f"Using {narrowest} posterior to calculate t0.")
        
        # Calculate it's median
        med = np.median(narrowest_posterior)
        
        # Use this to get the reference sample
        ii = np.argmin(np.abs(narrowest_posterior - med))

    # Get desired sample and return
    reference_parameters = {field:ref_pe_samples[field][ii] for field in parameter_names}
    return reference_parameters
    

def get_Tcut_from_Ncycles(waveform_dict, time_dict, ifo, Ncycles, ra, dec):
    
    """
    Compute the geocenter cutoff time corresponding to a given number of waveform cycles 
    after the peak amplitude.

    The function identifies extrema (peaks/troughs) in the detector strain data, locates 
    the absolute peak, and determines the time corresponding to the specified number of 
    half-cycles from the peak. If the target cycle lies between extrema, the function 
    linearly interpolates between them. The time is then shifted from the detector 
    reference frame to the geocenter frame using the given sky location.

    Parameters
    ----------
    waveform_dict : dict
        Dictionary mapping interferometer names (e.g., 'H1') to their strain time series (array-like).
    time_dict : dict
        Dictionary mapping interferometer names to their corresponding time arrays (same length as waveform).
    ifo : str
        Interferometer name for the detector in which we want to calculate the mapping from cycles to time. (
        Must be a key in waveform_dict and time_dict.
    Ncycles : float
        Number of full waveform cycles from the peak strain at which to compute the cutoff time.
        Negative Ncycles means before the peak, positive Ncycles means after the peak. 
    ra : float
        Right ascension used to transform from time in `ifo` to geocenter time
    dec : float
        Declination used to transform from time in `ifo` to geocenter time

    Returns
    -------
    tcut_geo : float
        Geocenter GPS time corresponding to the desired cutoff cycle.
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


def get_ACF(psd_dict, time_dict, f_low=11, f_max=None, nan_inf_replacement=1e10, patch=None, return_psds=False): 
   """
    Compute the autocorrelation function (ACF) from a PSD

    The PSDs are preprocessed by removing NaN/Inf values, clipping to Nyquist, patching values
    outside the frequency range of interest, and limiting the dynamic range. The ACF is computed 
    using the inverse real FFT of the PSD.

    Parameters
    ----------
    psd_dict : dict
        Dictionary mapping ifo names to their PSD arrays of shape (N, 2) 
        where the first column is frequency and the second is PSD value.
    time_dict : dict
        Dictionary mapping ifo names to their corresponding time arrays 
        (used to determine sampling rate and analysis length).
    f_low : float, optional
        Minimum frequency (Hz) to keep in the PSD before patching. Default is 11 Hz.
    f_max : float or None, optional
        Maximum frequency (Hz) to keep before patching. Defaults to Nyquist if None.
    nan_inf_replacement : float, optional
        Value to replace NaN or ±Inf entries in the PSD. Default is 1e10.
    patch : float or None, optional
        Value to patch PSD bins outside [f_low, f_max]. Default is None. 
        If None, uses 100 × max(PSD in range).
    return_psds : bool, optional
        If True, return the conditioned PSDs in addition to the ACF.

    Returns
    -------
    rho_dict : dict
        Dictionary mapping ifo names to their ACF arrays.
    cond_psds : dict, optional
        Dictionary mapping ifo names to their conditioned PSDs (only returned if return_psds=True).
    """
    
    rho_dict = OrderedDict()  # stores acf
    cond_psds = OrderedDict() # stores conditioned psds
    
    # Cycle through ifos
    for ifo, freq_psd in psd_dict.items():
                
        # unpack psd
        freq, psd = freq_psd.copy().T
        
        # Get dt
        times = time_dict[ifo]
        dt = times[1] - times[0]
        Nanalyze = len(times)
        
        # get rid of infs and nans and replace them with a large number
        psd = np.nan_to_num(
            psd, nan=nan_inf_replacement, posinf=nan_inf_replacement, neginf=nan_inf_replacement
        )

        # patch frequencies outside of bounds 
        nyquist_freq = 0.5 / dt
        if f_max is not None:
            if f_max > nyquist_freq:
                raise(ValueError, f"WARNING: f_max {f_max} cannot be greater than the nyquist frequency {nyquist_freq}")
        fmax = f_max if f_max is not None else nyquist_freq

        # get rid of everything above nyquist frequency
        m_ny = freq <= nyquist_freq
        freq = freq[m_ny]
        psd = psd[m_ny]

        # set values below f_low and greater than f_max to be equal to 100*max(psd)
        m_lower = freq >= f_low
        m_upper = freq <= fmax
        mask = m_lower & m_upper
        patch_ifo = 100 * max(psd[mask]) if patch is None else patch
        psd[~mask] = patch_ifo

        # check dynamic range -- aka look for giant spikes or blocks
        dynamic_range = max(np.log10(psd)) - min(np.log10(psd))
        if dynamic_range > 25: 
            print(f'alert! dynamic range of PSD is {int(dynamic_range)} orders of magnitude!')
            mask2 = np.log10(psd) > min(np.log10(psd)) + 25
            patch2 = 100*max(psd[~mask2])
            psd[mask2] = patch2

        # Store in dict
        cond_psds[ifo] = np.transpose([freq, psd])
        
        # Compute ACF from PSD
        rho = 0.5 * np.fft.irfft(psd) / dt  # dt comes from numpy fft conventions
        rho_dict[ifo] = rho[:Nanalyze]
        
    if return_psds: 
        return rho_dict, cond_psds
    else:
        return rho_dict
