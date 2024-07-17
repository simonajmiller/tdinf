import sys
import os
from time_domain_gw_inference import run_sampler, utils, group_postprocess
# import time_domain_gw_inference as td
import re
import numpy as np
import scipy
import pandas as pd
import multiprocessing as mp

import h5py
import argparse
import tqdm


def save_waveform_h5py(output_file, waveform_dict_list):
    """
    save list of waveform dicts
    """
    # Save output to HDF5
    with h5py.File(output_file, 'w') as f:
        for i, wf_dict in enumerate(waveform_dict_list):
            group = f.create_group(f'waveform_{i}')
            for key, value in wf_dict.items():
                group.create_dataset(key, data=value)


def load_waveform_h5py(output_file):
    # Load HDF5 file if it exists and return output
    with h5py.File(output_file, 'r') as f:
        waveform_dict_list = []
        for key in f.keys():
            group = f[key]
            wf_dict = {k: group[k][()] for k in group.keys()}
            waveform_dict_list.append(wf_dict)
    return waveform_dict_list


def compute_waveform(i, data_frame, likelihood_manager):
    params = data_frame.iloc[i]
    return group_postprocess.get_waveform_dict(params, likelihood_manager, ifos=likelihood_manager.ifos)


def load_or_get_N_waveforms(dataframe, likelihood_manager, output_file=None, overwrite=False, **kwargs):
    if not overwrite and output_file is not None and os.path.exists(output_file):
        # print('loading from existing file')
        return load_waveform_h5py(output_file)

    waveform_dict_list = group_postprocess.get_N_waveforms(dataframe, likelihood_manager, **kwargs)

    if output_file is not None:
        save_waveform_h5py(output_file, waveform_dict_list)
    return waveform_dict_list


def get_waveform_filename(directory, run_key):
    """ pass in directory where runs are stored, not the waveform directory"""
    waveform_dir = os.path.join(directory, 'waveforms')
    return os.path.join(waveform_dir, f'{run_key}_waveforms.h5')


if __name__ == "__main__":
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description="Script to save waveforms to file")

    # Add arguments
    parser.add_argument("--directory", type=str, help="Input directory")
    parser.add_argument("--run_key", type=str, help="name of run for which to create waveforms")

    parser.add_argument("--overwrite", action="store_true",
                        help="Flag to overwrite existing files (default: False)")
    parser.add_argument("--N_waveforms", type=int, default=300, help="Number of waveforms (default: 300)")
    parser.add_argument("--ncpu", type=int, default=mp.cpu_count(), help="Number of parallel processes to start")

    # Parse arguments
    args = parser.parse_args()

    # Assign arguments to variables
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
    full_args, \
    full_kwargs, \
    full_likelihood_manager = group_postprocess.get_settings_from_command_line_file(
        os.path.join(directory, 'command_line.sh'),
        filename_dict['full'],
        directory + '/',
        full_parser, verbose=True)

    sub_directory = filename_dict[args.run_key]
    dataframe = group_postprocess.load_dataframe(directory, sub_directory)

    waveform_dir = os.path.join(directory, 'waveforms')
    if not os.path.exists(waveform_dir):
        os.mkdir(waveform_dir)

    rand_ints = np.random.randint(len(dataframe), size=N_waveforms)

    delta_t = full_likelihood_manager.time_dict[full_likelihood_manager.ifos[0]][1] - \
              full_likelihood_manager.time_dict[full_likelihood_manager.ifos[0]][0]

    # Prepare the arguments for starmap
    parallel_args = [(i, dataframe, full_likelihood_manager) for i in rand_ints]

    with mp.Pool(processes=args.ncpu) as pool:
        # Use pool.starmap to parallelize the computation
        results = list(pool.starmap(compute_waveform, tqdm.tqdm(parallel_args, total=N_waveforms)))

    save_waveform_h5py(waveform_filename, results)
    print("all done! waveforms saved to:", waveform_filename)