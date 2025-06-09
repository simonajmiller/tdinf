import lal
import numpy as np
import h5py
from . import reconstructwf as rwf
from .spins_and_masses import *
import scipy.signal as sig
import json
import os


def parse_data_and_psds(args, initial_dir_path=''):
    """
    Convert command line arguments into dictionaries of paths to PSD and data.

    :param args: Command line arguments parsed by argparse.
    :return:
        data_path_dict: Dictionary mapping interferometer names to corresponding data paths.
        psd_path_dict: Dictionary mapping interferometer names to corresponding PSD paths.
    """
    def _split_ifo_from_arg_(argument, ifo, arg_name):
        prefix = f'{ifo}:'
        matching_paths = [os.path.join(initial_dir_path, path.replace(prefix, '')) for path in argument if path.startswith(prefix)]
        if not matching_paths:
            raise ValueError(
                f"Error: {ifo} {arg_name} not provided. "
                f"Either exclude that ifo or add --{arg_name} {ifo}:path/to/{arg_name}")
        if len(matching_paths) != 1:
            raise ValueError(
                f"Error: {ifo} {arg_name} was provided more than once! "
                f"Please only add --{arg_name} {ifo}:path/to/{arg_name} once")
        return matching_paths[0]

    data_path_dict = {}
    psd_path_dict = {}
    for ifo in args.ifos:
        data_path_dict[ifo] = _split_ifo_from_arg_(args.data, ifo, 'data')
        psd_path_dict[ifo] = _split_ifo_from_arg_(args.psd, ifo, 'psd')

    return data_path_dict, psd_path_dict


def hdf5_to_dict(hdf5_group):
    """
    Turn an hdf5 group into a dictionary for help loading in data
    """
    result_dict = {}
    for key, item in hdf5_group.items():
        if isinstance(item, h5py.Group):
            result_dict[key] = hdf5_to_dict(item)
        else:
            result_dict[key] = item[()]  # Convert dataset to NumPy array
    return result_dict


def load_raw_data(path_dict, ifos=('H1', 'L1', 'V1'), verbose=True):
    """
    Load in raw interferometer timeseries strain data
    
    Parameters
    ----------
    path_dict : dict
        dictionary:ifo_name -> to where each h5 file is located
    ifos : tuple of strings (optional)
        which interometers to load data from (some combination of 'H1', 'L1',
        and 'V1')
    verbose : boolean (optional)
        whether or not to print out information as the data is loaded
    
    Returns
    -------
    raw_time_dict : dictionary
        time stamps for the data from each ifo 
    raw_data_dict : dictionary
        the data from each ifo 
    """

    raw_time_dict = {}
    raw_data_dict = {}

    for ifo in ifos:

        try: 
            # for real data downloaded from gwosc...
            with h5py.File(path_dict[ifo], 'r') as f:
                strain = np.array(f['strain/Strain'])
                T0 = f['meta/GPSstart'][()]
                ts = T0 + np.arange(len(strain)) * f['meta/Duration'][()] / len(strain)
        except: 
            # for other data 
            with h5py.File(path_dict[ifo], 'r') as f:
                # Convert the HDF5 file to a dictionary
                data = hdf5_to_dict(f)
                strain = data['strain']
                ts = data['times']
                
        raw_time_dict[ifo] = ts
        raw_data_dict[ifo] = strain

        fsamp = 1.0 / (ts[1] - ts[0])
        if verbose:
            print("Raw %s data sampled at %.1f Hz" % (ifo, fsamp))

    return raw_time_dict, raw_data_dict


def get_pe_samples(path):
    """
    Load in parameter estimation (pe) samples from LVC GW190521 analysis, and calculate
    the peak strain time at geocenter and each detector, the detector antenna patterns,
    the psds, and the maximum posterior sky position

    Parameters
    ----------
    path : string (optional)
        file path for pe samples
    Returns
    -------
    pe_samples : dictionary
        parameter estimation samples released by the LVC
    """
    # Load in posterior samples
    with h5py.File(path, 'r') as f:
        try:
            try:
                pe_samples = f['NRSur7dq4']['posterior_samples'][()]
            except: 
                pe_samples = f['Exp0']['posterior_samples'][()]
        except:
            # hdf5 --> dict
            pe_samples_dict = hdf5_to_dict(f)['posterior']

            if isinstance(pe_samples_dict, np.ndarray):
                pe_samples = pe_samples_dict

            else:
                # dict --> a structured array with labels
                pe_samples = np.rec.fromarrays([pe_samples_dict[key] for key in pe_samples_dict],
                                               names=list(pe_samples_dict.keys()))
    return pe_samples


def parse_injected_parameters(filepath, initial_run_dir=None):
    """
    Function to load in the parameters for an injection
    """
    # Make sure we're passed a json file
    assert filepath[-4:] == 'json', 'File type not supported'

    json_file = os.path.join(initial_run_dir, filepath)

    # Load file 
    with open(json_file, 'r') as jf:
        inj_file = json.load(jf)

    # Format correctly
    injected_parameters = {p: inj_file[p] for p in inj_file.keys()}

    return injected_parameters


def load_posterior_samples(run_name, start_cut, end_cut, dx=0.5, pe_output_dir='../../data/output/',
                          prior_fname='prior.dat'): 
    
    """
    Function to load in posterior samples from one set of our runs
    """
    
    modes = ['pre', 'post']
    
    # Arange all the time slices to load
    cuts_float = np.arange(start_cut, end_cut + dx, dx)
    
    # Template for loading 
    if run_name=='':
        path_template = pe_output_dir + '{0}_{1}cycles.dat'
        cuts = cuts_float
        path_full = path_template.format('full', '0.0')
    else:
        path_template = pe_output_dir + f'{run_name}_' + '{0}_{1}cycles.dat'
        cuts = [int(c) if c.is_integer() else c for c in cuts_float]
        path_full = path_template.format('full', '0')

    # Dict for file paths 
    paths = {}

    # Cycle through the runs to get all the file paths
    for cut in cuts:
        for mode in modes:
            # Format the file name
            fname = path_template.format(mode, cut)

            # Add to paths fict
            key = f'{mode} {cut} cycles'
            paths[key] = fname

    # Samples from full duration (no time cut)
    paths['full'] = path_full

    # Prior samples
    paths['prior'] = pe_output_dir + prior_fname

    # Parse samples
    td_samples = {}
    for k, p in paths.items():

        # Check that the file exists 
        if os.path.exists(p):

            # Load
            samps = np.genfromtxt(p, names=True, dtype=float)

            # Calculate component masses
            m1s, m2s = m1m2_from_mtotq(samps['mtotal'], samps['q'])

            # Calculate chi-eff
            chieffs = chi_effective(m1s, samps['chi1'], samps['tilt1'], m2s, samps['chi2'], samps['tilt2'])

            # Calculate chi-p
            chips = chi_precessing(m1s, samps['chi1'], samps['tilt1'], m2s, samps['chi2'], samps['tilt2'])

            # Make into dict 
            samps_dict = {k: samps[k] for k in samps.dtype.names}
            samps_dict['m1'] = m1s
            samps_dict['m2'] = m2s
            samps_dict['chieff'] = chieffs
            samps_dict['chip'] = chips

            #  Add to over-all dict
            td_samples[k] = samps_dict

        else:
            print(f'could not find {p}')

    return td_samples
