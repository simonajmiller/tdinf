import numpy as np
import scipy.linalg as sl
from tqdm import tqdm

'''
Functions for whitening in the FREQUENCY DOMAIN 
'''

def whitenData(h_td, times, psd, psd_freqs, verbose=False):
    """
    Whiten a timeseries in the FREQUENCY DOMAIN with a given power spectral density
    
    Parameters
    ----------
    h_td : `numpy.array`
        un-whitened strain data in the time domain
    times : `numpy.array`
        timestamps of h_td
    psd : `numpy.array`
        power spectral density used to whiten the data at frequencies freqs
    psd_freqs : `numpy.array`
        frequencies corresponding to the psd
    
    Returns
    -------
    wh_td : `numpy.array`
        whitened time domain data at the same timestamps as the input
    """

    # Get segment length and frequencies
    dt = times[1] - times[0]
    Nt = len(h_td)
    freqs = np.fft.rfftfreq(Nt, dt)

    if verbose: 
        print(Nt, dt, freqs)

    # Interpolate PSD to the correct frequencies
    interp_psd = np.interp(freqs, psd_freqs, psd)

    # Into fourier domain
    h_fd = np.fft.rfft(h_td)

    # Divide out ASD and normalize properly 
    wh_fd = h_fd / (np.sqrt(interp_psd / dt / 2.))

    # Back into time domain
    wh_td = np.fft.irfft(wh_fd, n=Nt)

    return wh_td


def whiten_wfs(wf_dict_list, lm): 
    """
    Whiten a set of waveforms in the FREQUENCY DOMAIN given a likelihood 
    manager object `lm`.

    Parameters
    ----------
    wf_dict_list : list of dict
        List of waveform dictionaries, keyed by ifo name, with time series arrays.
    lm : LnLikelihoodManager object
        Likelihood manager containing:
            - time_dict: dict of time arrays
            - conditioned_psd_dict: dict of PSD arrays [freq, psd]
        See likelihood.LnLikelihoodManager.

    Returns
    -------
    list of dict
        Whitened waveform dictionaries.
    """
    wf_dict_list_wh = []
    for d in wf_dict_list: 
        d_wh = {ifo:whitenData(
            h_ifo, lm.time_dict[ifo], lm.conditioned_psd_dict[ifo][:,1], lm.conditioned_psd_dict[ifo][:,0]
        ) for ifo, h_ifo in d.items()}
        wf_dict_list_wh.append(d_wh)
    
    return wf_dict_list_wh


'''
Functions for whitening in the TIME DOMAIN
'''

def whitenData_TD(data, ACF):
    """
    Whiten data in the TIME DOMAIN using autocorrelation function (ACF).

    Parameters
    ----------
    data : np.ndarray
        Time series data.
    ACF : np.ndarray
        Autocorrelation function of the noise. Called rho in other
        parts of TDinf code.

    Returns
    -------
    np.ndarray
        Whitened data array.
    """
    # Get cholesky decomposition of Toeplitz matrix from ACF
    C = sl.toeplitz(ACF)
    L = sl.cholesky(C,lower=True)
    data_wh = sl.solve_triangular(L,data, lower=True)
    return data_wh


def whitenData_dict_TD(data_dict, lm):  
    """
    Whiten all ifo data in a dictionary in the TIME DOMAIN using `whitenData_TD`.

    Parameters
    ----------
    data_dict : dict
        Dictionary keyed by ifo containing time series arrays.
    lm : LnLikelihoodManager object
        Likelihood manager with `rho_dict` containing ACFs for each IFO.
        See likelihood.LnLikelihoodManager.

    Returns
    -------
    dict
        Whitened data dictionary.
    """
    return {ifo:whitenData_TD(data_dict[ifo], lm.rho_dict[ifo]) for ifo in lm.ifos}


def whiten_wfs_TD(wf_dict_list, L_dict): 
    """
    Whiten a set of waveforms in the TIME DOMAIN using precomputed 
    Cholesky factors.

    Parameters
    ----------
    wf_dict_list : list of dict
        List of waveform dictionaries keyed by IFO name.
    L_dict : dict
        Dictionary keyed by IFO containing lower-triangular Cholesky factors.

    Returns
    -------
    list of dict
        Whitened waveform dictionaries.
    """
    wf_dict_list_wh = []
    for d in tqdm(wf_dict_list): 
        d_wh = {ifo:sl.solve_triangular(L_dict[ifo], d[ifo], lower=True) for ifo in d.keys()}
        wf_dict_list_wh.append(d_wh)
    return wf_dict_list_wh