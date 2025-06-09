import numpy as np
from scipy.stats import gaussian_kde
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
import scipy.linalg as sl

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
    return x @ sl.solve_toeplitz(rho, y)

def calc_mf_SNR(d, s, rho):
    return inner_product(s, d, rho) / calc_opt_SNR(s, rho)

def calc_opt_SNR(s, rho): 
    return np.sqrt(inner_product(s, s, rho))

def calc_network_SNR(snr_list):
    snrs_sq = [snr ** 2 for snr in snr_list]
    network_snr = np.sqrt(sum(snrs_sq))
    return network_snr


"""
Other
"""

def interpolate_timeseries(time, values, new_time_grid):
    """
    Interpolates a timeseries to a new grid of points using cubic interpolation

    Parameters:
    - time: array-like, the original time grid
    - values: array-like, the values of the timeseries at the original time grid
    - new_time_grid: array-like, the new time grid

    Returns:
    - value_on_grid: array-like, the interpolated values at the new time grid
    """
    
    # cubic interp:
    values_interpolator = interp1d(time, values, kind='cubic', fill_value=0, bounds_error=False)
    value_on_grid = values_interpolator(new_time_grid)
    
    return value_on_grid

def apply_window(timeseries): 
    nsamps = len(timeseries)
    window = tukey(nsamps)
    window[int(0.5*nsamps):] = 1.
    return timeseries*window