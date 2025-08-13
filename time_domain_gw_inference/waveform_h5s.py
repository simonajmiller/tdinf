#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from time_domain_gw_inference import run_sampler, utils, group_postprocess
import re
import numpy as np
import scipy
import pandas as pd
import multiprocessing as mp
import h5py
import argparse
import tqdm

def save_waveform_h5py(output_file, waveform_dict_list, maxL_waveform_dict=None, time_dict=None):
    """
    Save waveforms into h5 object
    """
    # Save output to HDF5
    with h5py.File(output_file, 'w') as f:

        if time_dict is not None:
            # time stamps
            group = f.create_group('times')
            for key, value in time_dict.items():
                group.create_dataset(key, data=value) 

        if maxL_waveform_dict is not None:
            # maxL 
            group = f.create_group('maxL')
            for key, value in maxL_waveform_dict.items():
                group.create_dataset(key, data=value) 
        
        # random draws
        for i, wf_dict in enumerate(waveform_dict_list):
            group = f.create_group(f'waveform_{i}')
            for key, value in wf_dict.items():
                group.create_dataset(key, data=value)


def load_waveform_h5py(output_file):
    """
    Load waveform h5 file which was created with `save_waveform_h5py()`
    """
    # Load HDF5 file if it exists and return output
    with h5py.File(output_file, 'r') as f:
        waveforms = {}
        waveform_dict_list = []
        
        for key in f.keys():
            group = f[key]
            wf_dict = {k: group[k][()] for k in group.keys()}
            if key=='maxL' or key=='times': 
                waveforms[key] = wf_dict
            else:
                waveform_dict_list.append(wf_dict)
    
    waveforms['samples'] = waveform_dict_list    
    return waveforms


def get_waveform_dict(parameter, likelihood_manager, ifos=None):
    """
    Get the projected waveform for a given parameter point on a given 
    set of interferometers

    Parameters
    ----------
    parameter : dict
        Dictionary of physical parameters for the waveform.
    likelihood_manager : LnLikelihoodManager object
        Likelihood manager containing waveform manager and data info.
    ifos : list of str, optional
        List of detectors to return waveforms for. Defaults to all ifos in 
        likelihood_manager.

    Returns
    -------
    dict
        Dictionary of waveforms keyed by interferometer.
    """
    # if ifos not provided, default to all the ifos used stored in the 
    # likelihood_manager object
    if ifos is None:
        ifos = likelihood_manager.ifos

    # generated waveform for given parameter
    return likelihood_manager.waveform_manager.get_projected_waveform(
        parameter,
        ifos,
        time_dict=likelihood_manager.time_dict,
        f22_start=likelihood_manager.f22_start,
        f_ref=likelihood_manager.f_ref
    )


def compute_waveform(i, data_frame, likelihood_manager):
    """
    Compute the waveform for the i-th row of a dataframe.

    Parameters
    ----------
    i : int
        Row index in the dataframe.
    data_frame : pandas.DataFrame
        DataFrame containing physical parameter sets.
    likelihood_manager : LnLikelihoodManager object
        Likelihood manager containing waveform manager and data.

    Returns
    -------
    dict
        Projected waveform for the selected row parameters.
    """
    params = data_frame.iloc[i]
    return get_waveform_dict(params, likelihood_manager, ifos=likelihood_manager.ifos)


def get_waveform_filename(directory, run_key):
    """
    Construct the filename for storing waveform data.

    Parameters
    ----------
    directory : str
        Base directory where runs are stored (**not the waveform subdirectory**).
    run_key : str
        Key describing the run (e.g., 'pre_0.0' or 'full').

    Returns
    -------
    str
        Full path to the waveform h5 file.
    """
    waveform_dir = os.path.join(directory, 'waveforms')
    return os.path.join(waveform_dir, f'{run_key}_waveforms.h5')


def get_waveform_CI(wf_dict_list, lm): 
    '''
    Get the 90% credible region of strain from a list of waveforms
    `wf_dict_list` and a likelihood manager `lm`
    '''
    CIs_dict = {}
    for ifo in lm.ifos: 
        d_list = []
        for d in wf_dict_list: 
            d_list.append(d[ifo])
        d_arr = np.asarray(d_list)
        CIs = np.quantile(d_arr, (0.05, 0.5, 0.95), axis=0)
        CIs_dict[ifo]  = {'5th':CIs[0],'median':CIs[1],'95th':CIs[2]}
    return(CIs_dict)

def make_waveform_h5_arg_parser():
    
    parser = argparse.ArgumentParser(description="Script to save waveforms to file")

    # Add arguments
    parser.add_argument("--directory", type=str, help="Input directory")
    parser.add_argument("--run_key", type=str, help="name of run for which to create waveforms")

    parser.add_argument("--overwrite", action="store_true",
                        help="Flag to overwrite existing files (default: False)")
    parser.add_argument("--N_waveforms", type=int, default=300, help="Number of waveforms (default: 300)")
    parser.add_argument("--ncpu", type=int, default=mp.cpu_count(), help="Number of parallel processes to start")
    return parser


def main():
    # initialize ArgumentParser
    parser = make_waveform_h5_arg_parser()

    # parse arguments
    args = parser.parse_args()

    # assign arguments to variables
    directory = args.directory
    overwrite = args.overwrite
    N_waveforms = args.N_waveforms

    # don't overwrite existing files unless asked to explicitly
    waveform_filename = get_waveform_filename(directory, args.run_key)
    if not overwrite and waveform_filename is not None and os.path.exists(waveform_filename):
        # print('loading from existing file')
        print('waveform already exists, use --overwrite in order to overwrite all the files!')
        exit()

    # load full run (for the waveform generator)
    filename_dict = group_postprocess.generate_filename_dict(directory)
    full_parser = run_sampler.create_run_sampler_arg_parser()

    # load in the commandline file:
    if os.path.exists(os.path.join(directory, 'command_line.sh')):
        # if run with condor
        commandline_file = os.path.join(directory, 'command_line.sh') 
    else: 
        # if run with slurm 
        commandline_file = os.path.join(directory, 'tasks_run.txt')

    # parse commandline settings
    full_args, \
    full_kwargs, \
    full_likelihood_manager = group_postprocess.get_settings_from_command_line_file(
        commandline_file,
        filename_dict['full'],
        directory + '/',
        full_parser, 
        verbose=True
    )

    # bet relevant dataframe containing posterior samples
    sub_directory = filename_dict[args.run_key]
    dataframe = group_postprocess.load_dataframe(directory, sub_directory)

    # make waveform directory if it does not already exist
    waveform_dir = os.path.join(directory, 'waveforms')
    if not os.path.exists(waveform_dir):
        os.mkdir(waveform_dir)
        
    # find maxL parameters in posterior
    i_max_logL = np.argmax(dataframe['ln_posterior'] - dataframe['ln_prior'])
    max_logL_params = dataframe.iloc[i_max_logL]

    # generate waveform from maxL parameters
    maxL_wf_dict = full_likelihood_manager.waveform_manager.get_projected_waveform(
        max_logL_params,
        full_likelihood_manager.ifos, 
        time_dict = full_likelihood_manager.time_dict,
        f22_start = full_likelihood_manager.f22_start, 
        f_ref = full_likelihood_manager.f_ref
    )
        
    # generate N_waveforms random draws from the posterior
    if N_waveforms == len(dataframe): 
        print('generating reconstructions for all the waveforms')
        rand_ints = np.arange(N_waveforms)
    else:
        rand_ints = np.random.randint(len(dataframe), size=N_waveforms)

    # prepare the arguments for starmap
    parallel_args = [(i, dataframe, full_likelihood_manager) for i in rand_ints]

    # Uue pool.starmap to parallelize the computation
    with mp.Pool(processes=args.ncpu) as pool:
        results = list(
            pool.starmap(compute_waveform, tqdm.tqdm(parallel_args, total=N_waveforms))
        )

    # save result into h5 file
    save_waveform_h5py(
        waveform_filename, results, maxL_wf_dict, full_likelihood_manager.time_dict
    )
    print("all done! waveforms saved to:", waveform_filename)

if __name__ == "__main__":
    main()
