import numpy as np
import scipy
import lal
import lalsimulation
import os

import scipy.signal as sig
import sys
import gwpy

try:
    from spins_and_masses import transform_spins
except:
    from .spins_and_masses import transform_spins

lalsim = lalsimulation

def get_tgps_dict(tgps_geocent, ifos, ra, dec):
    """
     Get the time at each detector at the given geocenter time and sky position

     Parameters
     ----------
     tgps_geocent : float
         geocenter time
     ifos : tuple of strings (optional)
         which interometers to load data from (some combination of 'H1', 'L1',
         and 'V1')
     ra : float
         right ascension
     dec : float
         declination
     Returns
     -------
     tgps_dict : dictionary
         time at each detector at the given geocenter time and sky position
    """
    tgps_dict = {}

    # Cycle through interferometers
    for ifo in ifos:

        # Calculate time delay between geocenter and this ifo
        dt_ifo = lal.TimeDelayFromEarthCenter(lal.cached_detector_by_prefix[ifo].location,
                                              ra, dec, lal.LIGOTimeGPS(tgps_geocent))
        tgps_dict[ifo] = tgps_geocent + dt_ifo

    return tgps_dict


def get_antenna_pattern_dict(tgps_geocent, ifos, ra, dec, psi):
    """
     Get the antenna pattern at each detector at the given geocenter time and
     sky position

     Parameters
     ----------
     tgps_geocent : float
         geocenter time
     ifos : tuple of strings (optional)
         which interometers to load data from (some combination of 'H1', 'L1',
         and 'V1')
     ra : float
         right ascension
     dec : float
         declination
     psi : float
         polarization angle
     verbose : boolean (optional)
         whether or not to print out information calculated

     Returns
     -------
     ap_dict : dictionary
         antenna pattern for each interferometer at the given geocenter time and sky
         position
     """
    ap_dict = {}

    # Greenwich mean sidereal time
    gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(tgps_geocent))

    # Cycle through interferometers
    for ifo in ifos:
        # Calculate antenna pattern
        ap_dict[ifo] = lal.ComputeDetAMResponse(lal.cached_detector_by_prefix[ifo].response,
                                                ra, dec, psi, gmst)
    return ap_dict

def get_tgps_and_ap_dicts(tgps_geocent, ifos, ra, dec, psi):

    """
    Get the time and antenna pattern at each detector at the given geocenter time and
    sky position

    Parameters
    ----------
    tgps_geocent : float
        geocenter time
    ifos : tuple of strings (optional)
        which interometers to load data from (some combination of 'H1', 'L1',
        and 'V1')
    ra : float
        right ascension
    dec : float
        declination
    psi : float
        polarization angle

    Returns
    -------
    tgps_dict : dictionary
        time at each detector at the given geocenter time and sky position
    ap_dict : dictionary
        antenna pattern for each interferometer at the given geocenter time and sky
        position
    """
    return get_tgps_dict(tgps_geocent, ifos, ra, dec), get_antenna_pattern_dict(tgps_geocent, ifos, ra, dec, psi)