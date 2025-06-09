import numpy as np
import lal
from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions, SimInspiralTransformPrecessingWvf2PE

"""
Functions to calculate various mass quantities
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


"""
Functions to calculate various spin quantities
"""

def transformPrecessingWvf2PE( incl, S1x, S1y, S1z, S2x, S2y, S2z,  m1, m2, fRef, phiRef):
    """
    Get inclination angle and spin components at a given reference frequency from the
    masses, spin magnitudes, and various tilt and azimuthal angles
    Parameters
    ----------
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

    Returns
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
    """

    if isinstance(m1, float):
        return SimInspiralTransformPrecessingWvf2PE(incl, S1x, S1y, S1z, S2x, S2y, S2z, m1, m2, fRef, phiRef)

    thetaJN = np.zeros(len(m1))
    phiJL = np.zeros(len(m1))
    theta1 = np.zeros(len(m1))
    theta2 = np.zeros(len(m1))
    phi12 = np.zeros(len(m1))
    chi1 = np.zeros(len(m1))
    chi2 = np.zeros(len(m1))
    for i in range(len(m1)):
        thetaJN[i], phiJL[i], theta1[i], theta2[i], phi12[i], chi1[i], chi2[i] = SimInspiralTransformPrecessingWvf2PE(
            incl[i], S1x[i], S1y[i], S1z[i], S2x[i], S2y[i], S2z[i],  m1[i], m2[i], fRef[i], phiRef[i])

    return thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2


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
    m1_SI = m1 * lal.MSUN_SI
    m2_SI = m2 * lal.MSUN_SI

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
            incl[i], s1x[i], s1y[i], s1z[i], s2x[i], s2y[i], s2z[
                i] = SimInspiralTransformPrecessingNewInitialConditions(
                theta_jn[i], phi_jl[i], tilt1[i], tilt2[i], phi12[i], a1[i], a2[i],
                m1_SI[i], m2_SI[i], f_ref, phi_ref[i]
            )

    return incl, s1x, s1y, s1z, s2x, s2y, s2z


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

    q_inv = m1 / m2
    A1 = 2. + (3. * q_inv / 2.)
    A2 = 2. + 3. / (2. * q_inv)
    S1_perp = a1 * np.sin(tilt1) * m1 * m1
    S2_perp = a2 * np.sin(tilt2) * m2 * m2
    Sp = np.maximum(A1 * S2_perp, A2 * S1_perp)
    chi_p = Sp / (A2 * m1 * m1)
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

    chieff = (m1 * a1 * np.cos(tilt1) + m2 * a2 * np.cos(tilt2)) / (m1 + m2)
    return chieff