from . import run_sampler
from . import utils
import os
import pandas as pd
import re
from matplotlib.lines import Line2D
import imageio
import corner
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
import copy
from tqdm import tqdm

def load_run_settings_from_directory(directory, filename_dict=None, load_all_lm=False, verbose=False):
    """
    Given a directory with runs created via the pipeline, load in args, kwargs, likelihood_manager, and dataframe.
    
    Parameters
    ----------
    directory : str
        Directory containing runs.
    filename_dict : dict, optional
        Default is None. Dictionary describing the way that runs will be organized.
    load_all_lm : boolean, optional 
        Option to load ALL likelihood managers, not just that for the `full` run.
        Default = False.
    verbose : boolean, optional 
        Print out information as code is running. Helpful for, e.g., debugging.
        Default = False.
    
    Returns
    -------
    dict
        Dictionary with the following structure:
        settings = {
            'dir': directory,
            'runs': {
                'full': {
                    'likelihood_manager': ...,
                    'args': ...,
                    'kwargs': ...
                },
                ...
            }
            'dfs':{
                'full': ...
            }
        }
    """
    # Generate the filename dict if not provided by user
    if filename_dict is None:
        filename_dict = generate_filename_dict(directory)

    # Set up dict to return
    settings = {
        'dir': directory, 
        'runs': {
            key:{
                'args': None,'kwargs': None, 'likelihood_manager': None
            } for key in filename_dict.keys()
        },
        'dfs': {
            key: None for key in filename_dict.keys()
        }
    }

    # Make parser using run_sampler
    parser = run_sampler.create_run_sampler_arg_parser()

    # Cycle through runs
    for run in settings.keys():

        # Extract dir, runs, and dfs
        directory = settings['dir']
        td_settings = settings['runs']
        td_samples = settings['dfs']

        # Cycle through the runs
        for key in td_settings.keys():
            if verbose:
                print(f'key: {key}')

            # Whether or not to load likelihood manager
            load_lm = True if key == 'full' else load_all_lm 
            args_kwargs_only = not load_lm
            
            try:
                # if run with condor:
                if os.path.exists(os.path.join(directory, 'command_line.sh')): 
                    commandline_file = os.path.join(directory, 'command_line.sh') 
                # if run with slurm:
                else: 
                    commandline_file = os.path.join(directory, 'tasks_run.txt') 
                    
                # load output    
                output = get_settings_from_command_line_file(
                    commandline_file,
                    filename_dict[key],
                    directory + '/',
                    parser, 
                    verbose=verbose, 
                    args_kwargs_only=args_kwargs_only
                )

                # formatting info for the run
                if do_not_load_lm:
                    args, kwargs = output
                else:
                    args, kwargs, lm = output
                    td_settings[key]['likelihood_manager'] = lm
                td_settings[key]['args'] = args
                td_settings[key]['kwargs'] = kwargs
                
            except TypeError as e:
                print(e)
                print(f'unable to make {run} {key}')

            # load posterior into a DataFrame
            df = load_dataframe(directory, filename_dict[key])
            if df is None:
                print(f'WARNING: NO POSTERIOR FILE FOUND FOR {key} !!')
                continue
            td_samples[key] = df
            
    return settings


def generate_filename_dict(directory, pattern=r'(post|pre|full)_(-?\d+\.\d+)seconds'):
    """
    Generate a dictionary mapping descriptive keys to filenames in a directory
    that match a specific pattern.

    Parameters
    ----------
    directory : str
        Path to the directory containing files.
    pattern : str, optional
        Regular expression pattern to match filenames. Default matches
        'post', 'pre', or 'full' followed by a number and 'seconds'.

    Returns
    -------
    dict
        Dictionary mapping descriptive keys to filenames. 
        For 'full', the key is 'full'; for 'post' or 'pre', the key is
        formatted as 'post_<seconds>' or 'pre_<seconds>'.
    """
    
    filename_dict = {}
    pattern = re.compile(pattern)

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            prefix, nseconds = match.groups()
            if prefix == 'full':
                filename_dict[prefix] = filename
            else:
                filename_dict[f"{prefix}_{nseconds}"] = filename

    return filename_dict


def get_settings_from_command_line_file(
    command_line_file,
    file_prefix,
    initial_run_dir,
    parser,
    verbose=False,
    **kwargs
):
    """
    Read a command line string from a file containing all the run commands.

    This function searches a file for a line containing a specified prefix,
    then parses that line using the provided argument parser and returns
    the run settings.

    Parameters
    ----------
    command_line_file : str
        Path to the file containing command line strings for a set of runs.
    file_prefix : str
        Prefix string used to identify the relevant command line in the file.
    initial_run_dir : str
        Directory in which the paths in the command line are relative to.
    parser : argparse.ArgumentParser
        Parser created by create_run_sampler_arg_parser() to parse command line strings.
    verbose : bool, optional
        If True, print debug information. Default = False.
    **kwargs :
        Additional keyword arguments passed to get_settings_from_command_line_string().

    Returns
    -------
    tuple
        The result of get_settings_from_command_line_string() applied to the matching line.
    """
    
    # Open file containing command line run commands 
    with open(command_line_file, 'r') as f:
        # Read the lines
        lines = f.readlines()
        # Cycle through each line of the file
        for line in lines:
            # Search for the specified run tag (`file_prefix`) in that line
            if file_prefix in line:
                if verbose:
                    print("line is", line)
                # Rarse relevant commandline string
                return get_settings_from_command_line_string(
                    line,
                    initial_run_dir,
                    parser,
                    **kwargs
                )

def get_settings_from_command_line_string(
    command_line_string,
    initial_run_dir,
    parser,
    args_kwargs_only=False,
    return_ref_pe=False,
    verbose=False
):
    """
    Parse a command line string to extract run settings.

    This function takes a full command line string (as it would be entered in the shell),
    parses the arguments using the provided the run_sampler argument parser, and returns
    the run settings.

    Parameters
    ----------
    command_line_string : str
        Full command line string that was used to start the run.
    initial_run_dir : str
        Directory containing the initial run.
    parser : argparse.ArgumentParser
        Argument parser used to interpret the command line string.
    args_kwargs_only : bool, optional
        If True, only return args/kwargs and not also likelihood manager.
        Default = False.
    return_ref_pe : bool, optional
        If True, return reference posterior in addition to likelihood manager. 
        Default = False. 
    verbose : bool, optional
        If True, print additional debug information. Default = False.

    Returns
    -------
    dict
        Dictionary of settings extracted from the command line, including parsed
        arguments and any associated run objects.
    """
    
    # remove output redirection from the command line string
    if '&>>' in command_line_string:
        command_line_string = command_line_string.split('  &>>')[0]
    
    # skip the initial program name and parse remaining arguments
    skip_initial_arg = command_line_string.split()[1:]
    args = parser.parse_args(skip_initial_arg)
    
    # return settings based on parsed arguments
    return get_settings_from_args(
        args,
        initial_run_dir,
        verbose=verbose,
        args_kwargs_only=args_kwargs_only,
        return_ref_pe=return_ref_pe
    )


def get_settings_from_args(
    args,
    initial_run_dir,
    return_ref_pe=False,
    args_kwargs_only=False,
    verbose=False,
    custom_time_and_skypos=None
):
    """
    Extract run settings, initialize kwargs, and optionally create likelihood manager.

    This function processes parsed arguments, sets up reference parameters (including
    optional custom time and sky position), initializes kwargs, generates the waveform
    manager, conditions the data, and sets up a likelihood manager.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from create_run_sampler_arg_parser().
    initial_run_dir : str
        Directory in which the paths in args are relative to.
    return_ref_pe : bool, optional
        If True, return reference posterior samples along with likelihood manager.
        Default is False.
    args_kwargs_only : bool, optional
        If True, only return args/kwargs and not also likelihood manager.
        Default = False.
    verbose : bool, optional
        If True, print debug information. Default = False.
    custom_time_and_skypos : dict, optional
        Custom geocenter time and sky position for debugging. Keys should include
        'tgps_geocent', 'ra', 'dec', 'psi'. Default = None.

    Returns
    -------
    tuple
        if args_kwargs_only is True:
            (args, kwargs)
        elif return_ref_pe is False:
            (args, kwargs, likelihood_manager)
        else:
            (args, kwargs, likelihood_manager, reference_parameters, ref_pe_samples)
    """
    
    # Get reference parameters and reference posterior samples
    reference_parameters, ref_pe_samples = run_sampler.get_injected_parameters(
        args,
        initial_run_dir,
        verbose=verbose
    )

    # Override with custom time and sky position if provided
    if custom_time_and_skypos is not None:
        reference_parameters['geocent_time'] = custom_time_and_skypos['tgps_geocent']
        reference_parameters['geocenter_time'] = custom_time_and_skypos['tgps_geocent']
        reference_parameters['ra'] = custom_time_and_skypos['ra']
        reference_parameters['right_ascension'] = custom_time_and_skypos['ra']
        reference_parameters['dec'] = custom_time_and_skypos['dec']
        reference_parameters['declination'] = custom_time_and_skypos['dec']
        reference_parameters['psi'] = custom_time_and_skypos['psi']
        reference_parameters['polarization'] = custom_time_and_skypos['psi']

    # Initialize kwargs based on reference parameters
    kwargs = run_sampler.initialize_kwargs(args, reference_parameters)

    # If only loading args and kwargs, return early
    if args_kwargs_only:
        return args, kwargs

    # Create waveform manager
    if verbose:
        print('making waveform manager')
    wf_manager = run_sampler.make_waveform_manager(args, **kwargs)

    # Get conditioned data for likelihood
    if verbose:
        print('getting conditioned time and data')
    time_dict, data_dict, psd_dict = run_sampler.get_conditioned_time_and_data(
        args,
        wf_manager=wf_manager,
        reference_parameters=reference_parameters,
        initial_run_dir=initial_run_dir
    )

    if verbose:
        print("kwargs are:")
        print(kwargs)

    # Set up likelihood manager
    likelihood_manager = utils.LnLikelihoodManager(
        psd_dict=psd_dict,
        time_dict=time_dict,
        data_dict=data_dict,
        vary_time=args.vary_time,
        vary_skypos=args.vary_skypos,
        f_max=args.fmax,
        only_prior=args.only_prior,
        **kwargs
    )

    if return_ref_pe:
        return args, kwargs, likelihood_manager, reference_parameters, ref_pe_samples
    else:
        return args, kwargs, likelihood_manager
        

def load_dataframe(directory, run_directory_name):
    """
    Load a dataframe from a speficied run directory, and compute additional parameters.

    The function attempts to locate a `.dat` file with the posterior from a given run, 
    either directly in the specified directory or inside a subdirectory named after the
    run. Once loaded, additional parameters are computed and added to the dataframe.

    Parameters
    ----------
    directory : str
        Path to the directory containing the run data.
    run_directory_name : str
        Name of the run directory (or run file) to load.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with the loaded data and additional computed parameters.
        Returns None if the file could not be found or loaded.
    """

    # Set up the filename
    filename = os.path.join(directory, run_directory_name + '.dat')
    if not os.path.exists(filename):
        filename = os.path.join(directory, run_directory_name + '/' + run_directory_name + '.dat')
        
    # Try to load the file
    try:
        df = pd.read_csv(filename, delimiter='\s+')
    except:
        print(f'ERROR LOADING {filename}!!')
        return None
        
    # Calculate additional BBH parameters and return    
    return calc_additional_parameters(df)


def calc_additional_parameters(df):
    """
    Calculate additional BBH parameters of interest for a tdinf result: 
    - component masses (m1, m2)
    - chirp mass
    - effective inspiral spin 
    - effective precessing spin
    
    and wrap some angular quantities that are over-defined, to better match 
    convention in `bilby`
    - phase btwn 0 and 2pi
    - polarzation btwn 0 and pi
    
    Parameters
    ----------
    df : DataFrame
        Contains posterior directly output from tdinf
    
    Returns
    -------
    DataFrame
         df with additional columns with new parameters
    """

    # Calculate component masses
    m1s, m2s = utils.m1m2_from_mtotq(
        df['total_mass'], df['mass_ratio']
    )

    # Calculate chirp mass
    chirpmasses = utils.get_chirpmass(
        m1s, m2s
    )

    # Calculate chi-eff
    chieffs = utils.chi_effective(
        m1s, df['spin1_magnitude'], df['tilt1'], m2s, df['spin2_magnitude'], df['tilt2']
    )

    # Calculate chi-p
    chips = utils.chi_precessing(
        m1s, df['spin1_magnitude'], df['tilt1'], m2s, df['spin2_magnitude'], df['tilt2']
    )

    # Add to the df
    df['mass1'] = m1s
    df['mass2'] = m2s
    df['chirp_mass'] = chirpmasses
    df['chi_effective'] = chieffs
    df['chi_precessing'] = chips

    # fix phase wrapping
    df['phase'] = (df['phase'] + 2 * np.pi) % (2 * np.pi) 

    # fix polarization wrapping
    df['polarization'] = df['polarization'] % np.pi

    return df