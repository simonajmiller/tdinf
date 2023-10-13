import numpy as np
from scipy.stats import gaussian_kde
#import pycbc.psd
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design

"""
Logit transformations, used in likelihood
"""

def logit(x, xmin=0, xmax=1):
    return np.log(x - xmin) - np.log(xmax - x)

def inv_logit(y, xmin=0, xmax=1):
    return (np.exp(y)*xmax + xmin)/(1 + np.exp(y))

def logit_jacobian(x, xmin=0, xmax=1):
    return 1./(x-xmin) + 1./(xmax-x)


"""
Functions to calculate matched-filter SNR; 
See Eqs. (50) through (53) of  https://arxiv.org/pdf/2107.05609.pdf
"""

def inner_product(x, y, rho): 
    prod = x @ sl.solve_toeplitz(rho, y)
    return prod

def calc_mf_SNR(d, s, rho): 
    snr = inner_product(s, d, rho) / np.sqrt(inner_product(s, s, rho))
    return snr

def calc_network_mf_SNR(snr_list): 
    snrs_sq = [snr**2 for snr in snr_list]
    network_snr = np.sqrt(sum(snrs_sq))
    return network_snr


"""
Vector math 
"""

def get_mag(v): 
    
    """
    Get the magnitude of a vector v
    
    Parameters
    ----------
    v : `numpy.array`
        vector with components v[0], v[1], v[2], etc.
    
    Returns
    -------
    mag_v : float
        magnitude of v 
    """
    
    v_squared = [x*x for x in v]
    mag_v = np.sqrt(sum(v_squared))
    return mag_v


def unit_vector(v):
    
    """
    Get the unit of a vector v
    
    Parameters
    ----------
    v : `numpy.array`
        vector with components v[0], v[1], v[2], etc.
    
    Returns
    -------
    unit_v : `numpy.array`
        v divided by its magnitude
    """
    unit_v =  v / get_mag(v)
    return unit_v



"""
Other miscellaneous functions
"""

# def get_pycbc_PSD(filename, f_low, delta_f, sampling_freq=1024): 
    
#     """
#     Load in power spectral density from a file
    
#     Parameters
#     ----------
#     filename : string
#         path to the text file containing the psd
#     f_low : float
#         the lower frequency of the psd
#     delta_f : float
#         the frequency spacing of the psd
#     sampling_freq : float (optional)
#         the sampling frequency of the data the psd is for; defaults to 1024 Hz
        
#     Returns
#     -------
#     psd : pycbc.types.frequencyseries.FrequencySeries
#         the power spectral density as a pycbc frequency series 
#     """
    
#     # The PSD will be interpolated to the requested frequency spacing
#     length = int(sampling_freq / delta_f)
#     psd = pycbc.psd.from_txt(filename, length, delta_f, f_low, is_asd_file=False)
#     return psd


def bandpass(h, times, fmin, fmax):
    
    """
    Bandpass time-domain data between frequencies fmin and fmax
    
    Parameters
    ----------
    h: `numpy.array`
        strain data in the time domain
    times : `numpy.array`
        time stamps corresponding to h
    fmin : float
        minimum frequency (in Hertz) for bandpass filter
    fmas : float
        maximum frequency (in Hertz) for bandpass filte
    
    Returns
    -------
    h_hp : `numpy.array`
        bandpassed strain data at the same time stamps as h 
    """
    
    # turn into gwpy TimeSeries object so we can use the built in filtering functions
    h_timeseries = TimeSeries(h, t0=times[0], dt=times[1]-times[0])
    
    # design the bandpass filter we want
    bp_filter = filter_design.bandpass(fmin, fmax, h_timeseries.sample_rate)
    
    # filter the timeseries
    h_bp = h_timeseries.filter(bp_filter, filtfilt=True)
    
    return h_bp

def reflected_kde(samples, lower_bound, upper_bound, npoints=500, bw=None): 
    
    """
    Generate a ONE DIMENSIONAL reflected Gaussian kernal density estimate (kde) 
    for the input samples, bounded between lower_bound and upper_bound
    
    Parameters
    ----------
    samples : `numpy.array`
        datapoints to estimate the density from
    lower_bound : float
        lower bound for the reflection
    upper_bound : float
        upper bound for the reflection
    npoints : int or `numpy.array` (optional)
        if int, number of points on which to calculate grid; if array, the
        grid itself (or any set of points on which to evaluate the samples)
    bw : str, scalar or callable (optional)
        the method used to calculate the estimator bandwidth; if None, defaults to
        'scott' method. see documentation here:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    
    Returns
    -------
    grid : `numpy.array`
        points on which kde is evaluated 
    kde_on_grid : `numpy.array`
        reflected kde evaluated on the points in grid
    """
    
    if isinstance(npoints, int):
        grid = np.linspace(lower_bound, upper_bound, npoints)
    else:
        grid = npoints
    
    kde_on_grid = gaussian_kde(samples, bw_method=bw)(grid) + \
                  gaussian_kde(2*lower_bound-samples, bw_method=bw)(grid) + \
                  gaussian_kde(2*upper_bound-samples, bw_method=bw)(grid) 
    
    return grid, kde_on_grid