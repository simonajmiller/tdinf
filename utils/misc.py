import numpy as np
from scipy.stats import gaussian_kde
import lal
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
import pycbc.psd
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design

"""
Functions to calculate various spin quantities
"""

def chi_precessing(m1, a1, tilt1, m2, a2, tilt2):
    
    """
    Calculate the effective precessing spin, chi_p
    
    Parameters
    ----------
    m1 : `numpy.array` or float
        primary mass
    a1 : `numpy.array` or float
        spin magnitude of the primary mass
    tilt1 : `numpy.array` or float
        tilt angle (in radians) of the primary mass
    m2 : `numpy.array` or float
        secondary mass (m2 <= m1)
    a2 : `numpy.array` or float
        spin magnitude of the secondary mass
    tilt2 : `numpy.array` or float
        tilt angle (in radians) of the secondary mass
    
    Returns
    -------
    chi_p : `numpy.array` or float
        effective precessing spin 
    """
    
    q_inv = m1/m2
    A1 = 2. + (3.*q_inv/2.)
    A2 = 2. + 3./(2.*q_inv)
    S1_perp = a1*np.sin(tilt1)*m1*m1
    S2_perp = a2*np.sin(tilt2)*m2*m2
    Sp = np.maximum(A1*S2_perp, A2*S1_perp)
    chi_p = Sp/(A2*m1*m1)
    return chi_p


def chi_effective(m1, a1, tilt1, m2, a2, tilt2):
    
    """
    Calculate the effective spin, chi_eff
    
    Parameters
    ----------
    m1 : `numpy.array` or float
        primary mass
    a1 : `numpy.array` or float
        spin magnitude of the primary mass
    tilt1 : `numpy.array` or float
        tilt angle (in radians) of the primary mass
    m2 : `numpy.array` or float
        secondary mass (m2 <= m1)
    a2 : `numpy.array` or float
        spin magnitude of the secondary mass
    tilt2 : `numpy.array` or float
        tilt angle (in radians) of the secondary mass
    
    Returns
    -------
    chi_eff : `numpy.array` or float
        effective spin 
    """
    
    chieff = (m1*a1*np.cos(tilt1) + m2*a2*np.cos(tilt2))/(m1+m2)
    return chieff


def calculate_generalizedChiP(m1, a1, tilt1, m2, a2, tilt2, phi12): 
    
    """
    Calculate generalized chi_p: Eq.(15) of https://arxiv.org/abs/2011.11948
    
    Parameters
    ----------
    m1 : `numpy.array` or float
        primary mass
    a1 : `numpy.array` or float
        spin magnitude of the primary mass
    tilt1 : `numpy.array` or float
        tilt angle (in radians) of the primary mass
    m2 : `numpy.array` or float
        secondary mass (m2 <= m1)
    a2 : `numpy.array` or float
        spin magnitude of the secondary mass
    tilt2 : `numpy.array` or float
        tilt angle (in radians) of the secondary mass
    
    Returns
    -------
    gen_chip : `numpy.array` or float
        generalized chi_p 
    """
    
    q = m2/m1
    omega_tilda = q*(4*q + 3)/(4 + 3*q) 
    
    term1 = a1*np.sin(tilt1)
    term2 = omega_tilda*a2*np.sin(tilt2)
    term3 = 2*omega_tilda*a1*a2*np.sin(tilt1)*np.sin(tilt2)*np.cos(phi12)
    
    gen_chip = np.sqrt(term1**2 + term2**2 + term3**3)
    
    return gen_chip


def calculate_magnitudeChiPerp(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z): 
    
    """
    Calculate magnitude of chi_perp: Eq.(9) of https://arxiv.org/abs/2012.02209
    
    Parameters
    ----------
    m1 : `numpy.array` or float
        primary mass
    m2 : `numpy.array` or float
        secondary mass (m2 <= m1)
    s1x : `numpy.array` or float
        x-component spin of primary mass
    s1y : `numpy.array` or float
        y-component spin of primary mass
    s1z : `numpy.array` or float
        z-component spin of primary mass
    s2x : `numpy.array` or float
        x-component spin of secondary mass
    s2y : `numpy.array` or float
        y-component spin of secondary mass
    s2z : `numpy.array` or float
        z-component spin of secondary mass
    
    Returns
    -------
    mag_chiperp : `numpy.array` or float
        magnitude of the chi_perp vector 
    """

    # dimensonless spin vectors
    vec_chi1 = [s1x, s1y, s1z]
    vec_chi2 = [s2x, s2y, s2z]
    
    # add dimension
    vec_S1 = np.array(vec_chi1)*m1*m1
    vec_S2 = np.array(vec_chi2)*m2*m2
    
    # total spin 
    vec_S = vec_S1 + vec_S2 
        
    # mag of in plane (perp) components
    S1_perp = get_mag(vec_S1[0:-1])
    S2_perp = get_mag(vec_S2[0:-1])
    S_perp = get_mag(vec_S[0:-1])
    
    # figured out norm based on conditions
    mask = S1_perp>=S2_perp
    norm1 = m1*m1 + S2_perp
    norm2 = m2*m2 + S1_perp
        
    # calculate mag_ chiperp
    mag_chiperp = np.zeros(len(m1))
    mag_chiperp[mask] = S_perp[mask]/norm1[mask]
    mag_chiperp[~mask] = S_perp[~mask]/norm2[~mask]
    
    return mag_chiperp


def transform_spins(theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1, m2, f_ref, phi_ref):
    
    """
    Get inclination angle and spin components at a given reference frequency from the 
    masses, spin magnitudes, and various tilt and azimuthal angles
    
    Parameters
    ----------
    theta_jn : `numpy.array` or float
        zenith angle (in radians) between J (total angular momentum) and N (line of sight)
    phi_jl : `numpy.array` or float
        azimuthal angle (in radians) of L_N (orbital angular momentum) on its cone about J
    tilt1 : `numpy.array` or float
        tilt angle (in radians) of the primary mass
    tilt2 : `numpy.array` or float
        tilt angle (in radians) of the secondary mass
    phi12 : `numpy.array` or float
        azimuthal angle  (in radians) between the projections of the component spins onto 
        the orbital plane
    a1 : `numpy.array` or float
        spin magnitude of the primary mass
    a2 : `numpy.array` or float
        spin magnitude of the secondary mass
    m1 : `numpy.array` or float
        primary mass in solar masses
    m2 : `numpy.array` or float
        secondary mass (m2 <= m1) in solar masses
    f_ref : float
        reference frequency (in Hertz)
    phi_ref : `numpy.array` or float
        reference phase (in radians) 
    
    Returns
    -------
    iota : `numpy.array` or float 
        inclination angle of the binary at f_ref
    s1x : `numpy.array`  or float
        x-component spin of primary mass at f_ref
    s1y : `numpy.array` or float
        y-component spin of primary mass at f_ref
    s1z : `numpy.array` or float
        z-component spin of primary mass at f_ref
    s2x : `numpy.array` or float
        x-component spin of secondary mass at f_ref
    s2y : `numpy.array` or float
        y-component spin of secondary mass at f_ref
    s2z : `numpy.array` or float
        z-component spin of secondary mass at f_ref
    """

    # Transform spins 
    m1_SI = m1*lal.MSUN_SI   
    m2_SI = m2*lal.MSUN_SI
    
    # Check if float or array 
    if isinstance(m1, float):
        incl, s1x, s1y, s1z, s2x, s2y, s2z = SimInspiralTransformPrecessingNewInitialConditions(
            theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1_SI, m2_SI, f_ref, phi_ref
        )
    else:
        nsamps = len(m1)
        incl = np.zeros(nsamps)
        s1x = np.zeros(nsamps)
        s1y = np.zeros(nsamps)
        s1z = np.zeros(nsamps)
        s2x = np.zeros(nsamps)
        s2y = np.zeros(nsamps)
        s2z = np.zeros(nsamps)

        for i in range(nsamps): 

            incl[i], s1x[i], s1y[i], s1z[i], s2x[i], s2y[i], s2z[i] = SimInspiralTransformPrecessingNewInitialConditions(
                theta_jn[i], phi_jl[i], tilt1[i], tilt2[i], phi12[i], a1[i], a2[i], 
                m1_SI[i], m2_SI[i], f_ref, phi_ref[i]
            )
        
    return incl, s1x, s1y, s1z, s2x, s2y, s2z


"""
Functions to whiten and bandpass the data
"""

def get_pycbc_PSD(filename, f_low, delta_f, sampling_freq=1024): 
    
    """
    Load in power spectral density from a file
    
    Parameters
    ----------
    filename : string
        path to the text file containing the psd
    f_low : float
        the lower frequency of the psd
    delta_f : float
        the frequency spacing of the psd
    sampling_freq : float (optional)
        the sampling frequency of the data the psd is for; defaults to 1024 Hz
        
    Returns
    -------
    psd : pycbc.types.frequencyseries.FrequencySeries
        the power spectral density as a pycbc frequency series 
    """
    
    # The PSD will be interpolated to the requested frequency spacing
    length = int(sampling_freq / delta_f)
    psd = pycbc.psd.from_txt(filename, length, delta_f, f_low, is_asd_file=False)
    return psd


def whitenData(h_td, psd, freqs):
    
    """
    Whiten a timeseries with a given power spectral density
    
    Parameters
    ----------
    h_td : `numpy.array`
        un-whitened strain data in the time domain
    psd : `numpy.array`
        power spectral density used to whiten the data at frequencies freqs
    freqs : `numpy.array`
        frequencies corresponding to the psd
    
    Returns
    -------
    wh_td : `numpy.array`
        whitened time domain data at the same timestamps as the input
    """
    
    # Get segment length and sampling rate 
    dt = 0.5 / round(freqs.max())
    df = freqs[1] - freqs[0]
    seglen = 1 / df
    sampling_rate = 1 / dt 
    N = int(seglen * sampling_rate) - 1
    
    # Into fourier domain
    h_fd = np.fft.rfft(h_td, n=N) / N
    
    # Divide out ASD 
    wh_fd = h_fd/np.sqrt(psd * seglen / 4)
    
    # Back into time domain
    wh_td = 0.5*np.fft.irfft(wh_fd) / dt
    wh_td = wh_td[:len(h_td)]
    
    return wh_td


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
Other miscellaneous functions
"""

def m1m2_from_mtotq(mtot, q):
    
    """
    Calculate component masses from total mass and mass ratio
    
    Parameters
    ----------
    mtot : float or `numpy.array`
        total mass
    q : float or `numpy.array`
        mass ratio (q <= 1)
    
    Returns
    -------
    m1 : float or `numpy.array`
        primary mass
    m2 : float or `numpy.array`
        secondary mass (m2 <= m1)
    """
    m1 = mtot / (1 + q)
    m2 = mtot - m1
    return m1, m2


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