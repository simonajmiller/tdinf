import numpy as np

def whitenData(h_td, times, psd, psd_freqs):
    
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
    
    # Interpolate PSD to the correct frequencies
    interp_psd = np.interp(freqs, psd_freqs, psd)
    
    # Into fourier domain
    h_fd = np.fft.rfft(h_td)
    
    # Divide out ASD and normalize properly 
    wh_fd = h_fd / (np.sqrt(interp_psd /dt/2.))
    
    # Back into time domain
    wh_td = np.fft.irfft(wh_fd, n=Nt)
    
    return wh_td