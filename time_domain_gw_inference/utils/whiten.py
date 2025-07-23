import numpy as np

def whitenData(h_td, times, psd, psd_freqs, verbose=False):
    """
    Whiten a timeseries with a given power spectral density
    
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
    '''
    Whiten a set of waveforms (wf_dict_list) given a likelihood manager object (lm)
    '''
    wf_dict_list_wh = []
    for d in wf_dict_list: 
        d_wh = {ifo:whitenData(
            h_ifo, lm.time_dict[ifo], lm.conditioned_psd_dict[ifo][:,1], lm.conditioned_psd_dict[ifo][:,0]
        ) for ifo, h_ifo in d.items()}
        wf_dict_list_wh.append(d_wh)
    
    return wf_dict_list_wh