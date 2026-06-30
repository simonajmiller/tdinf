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
    """
    Apply generalized logit transformation.

    Maps a value from the open interval (xmin, xmax) to the real line (-∞, ∞).
    For xmin=0 and xmax=1, this reduces to the standard logit transform:
        logit(x) = log(x / (1 - x))

    Parameters
    ----------
    x : array_like
        Input value(s) in the range (xmin, xmax).
    xmin : float, optional
        Lower bound of the transformation domain. Default is 0.
    xmax : float, optional
        Upper bound of the transformation domain. Default is 1.

    Returns
    -------
    ndarray or float
        Transformed value(s) on the real line.

    Notes
    -----
    Formula:
        logit(x) = log(x - xmin) - log(xmax - x)
    """
    return np.log(x - xmin) - np.log(xmax - x)

def inv_logit(y, xmin=0, xmax=1):
    """
    Apply the inverse of the generalized logit transformation.

    Maps a value from the real line (-∞, ∞) back to the open interval (xmin, xmax).
    For xmin=0 and xmax=1, this reduces to the standard inverse logit (sigmoid) transform.

    Parameters
    ----------
    y : array_like
        Input value(s) on the real line.
    xmin : float, optional
        Lower bound of the output range. Default is 0.
    xmax : float, optional
        Upper bound of the output range. Default is 1.

    Returns
    -------
    ndarray or float
        Transformed value(s) in the range (xmin, xmax).

    Notes
    -----
    Formula:
        inv_logit(y) = (exp(y) * xmax + xmin) / (1 + exp(y))
    """
    return (np.exp(y)*xmax + xmin)/(1 + np.exp(y))

def logit_jacobian(x, xmin=0, xmax=1):
    """
    Compute the derivative (Jacobian) of the generalized logit transformation.

    This is the derivative of `logit(x, xmin, xmax)` with respect to `x`.

    Parameters
    ----------
    x : array_like
        Input value(s) in the range (xmin, xmax).
    xmin : float, optional
        Lower bound of the transformation domain. Default is 0.
    xmax : float, optional
        Upper bound of the transformation domain. Default is 1.

    Returns
    -------
    ndarray or float
        The Jacobian (first derivative) evaluated at `x`.

    Notes
    -----
    Formula:
        d/dx logit(x) = 1 / (x - xmin) + 1 / (xmax - x)
    """
    return 1./(x-xmin) + 1./(xmax-x)


"""
Functions to calculate matched-filter SNR; 
See Eqs. (50) through (53) of  https://arxiv.org/pdf/2107.05609.pdf
"""

def inner_product(x, y, rho):
    """
    Compute the noise-weighted inner product between two signals.

    The inner product is defined as:
        ⟨x, y⟩ = xᵀ C⁻¹ y
    where C is the covariance matrix of the noise, represented here by
    its first row `rho` (Toeplitz structure), which is the ACF.

    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector.
    rho : array_like
        ACF, aka first row of the noise covariance matrix (Toeplitz form).

    Returns
    -------
    float
        The noise-weighted inner product ⟨x, y⟩.
    """
    return x @ sl.solve_toeplitz(rho, y)

def inner_product_gs(x, y, T_inv_first_col):
    """
    Compute the noise-weighted inner product using Gohberg-Semencul formula.

    This function computes the inner product ⟨x, y⟩ = xᵀ C⁻¹ y using the
    Gohberg-Semencul formula for efficient multiplication by the inverse of a
    symmetric Toeplitz matrix. The input `T_inv_first_col` is the first column
    of C⁻¹, which can be computed using Levinson-Durbin algorithm.

    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector.
    T_inv_first_col : array_like
        The first column of the inverse symmetric Toeplitz matrix C⁻¹.

    Returns
    -------
    float
        The noise-weighted inner product ⟨x, y⟩.
    """
    return x @ gohberg_semencul_multiply(T_inv_first_col, y)

def gohberg_semencul_prep(t_col):
    """
    Preparation to calculate using Gohberg-Semencul formula.
    Computes the first column of T^{-1} given the first column of T,
    assuming T is a symmetric Toeplitz matrix.

    Parameters:
    t_col : ndarray
        The first column of the symmetric Toeplitz matrix T.
    Returns:
    x : ndarray
        The first column of the inverse Toeplitz matrix T^{-1}.
    """
    e0 = np.zeros(len(t_col))
    e0[0] = 1.0
    x = sl.solve_toeplitz((t_col, t_col), e0)
    return x

def gohberg_semencul_multiply(Tinv_first_col, b):
    """
    Calculate the product T^{-1} b using the Gohberg-Semencul formula.

    Parameters:
    Tinv_first_col : ndarray
        The first column of the inverse symmetric Toeplitz matrix T^{-1}.
        Can be computed using Levinson-Durbin algorithm.
        See gohberg_semencul_prep below.
    b : ndarray
        The vector to be multiplied by T^{-1}.
    """
    N = len(Tinv_first_col)
    x0 = Tinv_first_col[0]

    c_L1 = Tinv_first_col
    # c_L1[0] = 1
    r_L1 = np.zeros(N)
    r_L1[0] = Tinv_first_col[0]  # First row is [x0, 0, ..., 0]

    # L2 is Lower Toeplitz defined by reversed shift of x
    # Column is [0, x_{N-1}, x_{N-2}, ..., x_1]
    c_L2 = np.concatenate(([0], Tinv_first_col[1:][::-1]))
    r_L2 = np.zeros(N)  # First row is all zeros

    # 3. Compute Matrix-Vector Products using matmul_toeplitz
    # Note: matmul_toeplitz((c, r), v) computes T(c,r) @ v
    # To compute transpose T(c,r).T @ v, we simply swap inputs: T(r,c) @ v

    # Term 1: L1 @ (L1.T @ b)
    # Inner: L1.T @ b -> use (r_L1, c_L1)
    inner_1 = sl.matmul_toeplitz((r_L1, c_L1), b)
    # Outer: L1 @ inner -> use (c_L1, r_L1)
    term_1 = sl.matmul_toeplitz((c_L1, r_L1), inner_1)

    # Term 2: L2 @ (L2.T @ b)
    # Inner: L2.T @ b -> use (r_L2, c_L2)
    inner_2 = sl.matmul_toeplitz((r_L2, c_L2), b)
    # Outer: L2 @ inner -> use (c_L2, r_L2)
    term_2 = sl.matmul_toeplitz((c_L2, r_L2), inner_2)

    # 4. Combine
    return (1.0 / x0) * (term_1 - term_2)


def calc_mf_SNR(d, s, rho):
    """
    Calculate the matched-filter signal-to-noise ratio (SNR).

    The matched-filter SNR is given by:
        ρ_MF = ⟨s, d⟩ / √⟨s, s⟩
    where `d` is the data, `s` is the template signal, and the inner
    product is noise-weighted.

    Parameters
    ----------
    d : array_like
        Observed data.
    s : array_like
        Template signal.
    rho : array_like
        First row of the noise covariance matrix (Toeplitz form).

    Returns
    -------
    float
        The matched-filter SNR.
    """
    return inner_product(s, d, rho) / calc_opt_SNR(s, rho)


def calc_opt_SNR(s, rho): 
    """
    Calculate the optimal signal-to-noise ratio (SNR) for a template.

    The optimal SNR is defined as:
        ρ_opt = √⟨s, s⟩
    where the inner product is noise-weighted.

    Parameters
    ----------
    s : array_like
        Template signal.
    rho : array_like
        First row of the noise covariance matrix (Toeplitz form).

    Returns
    -------
    float
        The optimal SNR.
    """
    return np.sqrt(inner_product(s, s, rho))


def calc_network_SNR(snr_list):
    """
    Calculate the combined network SNR from individual detector SNRs.

    The network SNR is computed as:
        ρ_net = √(∑ ρ_i²)
    where ρ_i is the SNR from the i-th detector.

    Parameters
    ----------
    snr_list : list of float
        List of individual detector SNR values.

    Returns
    -------
    float
        The combined network SNR.
    """
    snrs_sq = [snr ** 2 for snr in snr_list]
    network_snr = np.sqrt(sum(snrs_sq))
    return network_snr


"""
Other
"""

def interpolate_timeseries(time, values, new_time_grid):
    """
    Interpolate a time series to a new grid using cubic interpolation.

    This function constructs a cubic spline interpolator for the given
    time series and evaluates it on the specified `new_time_grid`.
    Values outside the original `time` range are set to 0.

    Parameters
    ----------
    time : array_like
        Original time grid (1D, increasing).
    values : array_like
        Values of the time series at the original time grid.
    new_time_grid : array_like
        New time grid where the time series should be interpolated.

    Returns
    -------
    ndarray
        Interpolated values of the time series on `new_time_grid`.

    Notes
    -----
    Uses `scipy.interpolate.interp1d` with:
        kind='cubic', fill_value=0, bounds_error=False
    """
    values_interpolator = interp1d(time, values, kind='cubic', fill_value=0, bounds_error=False)
    value_on_grid = values_interpolator(new_time_grid)
    
    return value_on_grid


def apply_window(timeseries, alpha=0.5):
    """
    Apply a Tukey window to the first portion of a time series.

    A Tukey window is generated for the full series length, then modified so
    that all samples after index `alpha * nsamps` are set to 1 (no tapering).
    The resulting window is multiplied element-wise with the input time series.

    Parameters
    ----------
    timeseries : array_like
        Input time series to be windowed.
    alpha : float, optional
        Fraction of the series length over which the Tukey taper is applied.
        Must be between 0 and 1. The default is 0.5, meaning the first half
        is tapered and the second half remains unchanged.

    Returns
    -------
    ndarray
        Windowed time series.

    Notes
    -----
    The Tukey window shape parameter is left at its default (alpha=0.5 in
    `scipy.signal.windows.tukey`). The modification here controls where the
    taper stops, not the Tukey shape parameter.
    """
    nsamps = len(timeseries)
    window = tukey(nsamps)
    window[int(alpha*nsamps):] = 1.
    return timeseries*window