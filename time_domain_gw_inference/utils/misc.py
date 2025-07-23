import numpy as np
from scipy.stats import gaussian_kde
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
import scipy.linalg as sl
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from collections import namedtuple
try:
    from scipy.signal import tukey
except ImportError:
    from scipy.signal.windows import tukey

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

# I've copied the code used to calculate JS divergences in bilby review from 
# https://git.ligo.org/gregory.ashton/bilby_mcmc_validation/-/blob/master/bilby_test/bilby_test/utils.py

def calc_median_error(jsvalues, quantiles=(0.16, 0.84)):
    quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
    quants = np.percentile(jsvalues, quants_to_compute * 100)
    summary = namedtuple("summary", ["median", "lower", "upper"])
    summary.median = quants[1]
    summary.plus = quants[2] - summary.median
    summary.minus = summary.median - quants[0]
    return summary

def calculate_js(samplesA, samplesB, weightsA=None, weightsB=None, ntests=100, xsteps=100, nsamples=None):
    js_array = np.zeros(ntests)
    for j in range(ntests):
        if nsamples is None:
            nsamples = min([len(samplesA), len(samplesB)])
        A = np.random.choice(samplesA, size=nsamples, replace=False, p=weightsA)
        B = np.random.choice(samplesB, size=nsamples, replace=False, p=weightsB)
        xmin = np.min([np.min(A), np.min(B)])
        xmax = np.max([np.max(A), np.max(B)])
        x = np.linspace(xmin, xmax, xsteps)
        A_pdf = gaussian_kde(A)(x)
        B_pdf = gaussian_kde(B)(x)

        js_array[j] = np.nan_to_num(np.power(jensenshannon(A_pdf, B_pdf), 2))

    return calc_median_error(js_array)