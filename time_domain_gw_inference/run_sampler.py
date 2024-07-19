#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import emcee
import json
import pandas as pd
from multiprocessing import Pool
from contextlib import closing
import os
import time_domain_gw_inference.utils as utils


def create_run_sampler_arg_parser():
    """
    Parse arguments
    """
    p = argparse.ArgumentParser()

    # Path to where to save data 
    p.add_argument('-o', '--output-h5', help='h5 filename for emcee', required=True)

    # Whether to run pre-Tcut, post-Tcut, or full (Tstart to Tend)?
    p.add_argument('-m', '--mode', required=True)

    # Args for cutoff (defined in # of cycles OR seconds from merger; up to user)
    p.add_argument('-t', '--Tcut-cycles', type=float, default=None)
    p.add_argument('-ts', '--Tcut-seconds', type=float, default=None)

    # Start & end times for segment of data to analyze
    p.add_argument('--Tstart', type=float, default=1242442966.9077148)
    p.add_argument('--Tend', type=float, default=1242442967.607715)

    # Place where input data is stored
    p.add_argument('--pe-posterior-h5-file', default=None,
                   help='posterior file containing pe samples, used only if injected-parameters==None')
    p.add_argument('--data', default=None, required=True,
                   help='path to data formatted as --data {ifo}:path/To/psd', action='append')
    p.add_argument('--psd', default=None, required=True,
                   help='path to data formatted as --psd {ifo}:path/To/psd', action='append')

    # Option to do an injection instead of use real data;
    p.add_argument('--injected-parameters', default=None)
    p.add_argument('--reference-parameters', default=None, help='json of parameters that initialize '
                                                                '1) how 0 is defined in the time cuts '
                                                                '2) the initialization of prior draw points ')

    # Optional args for waveform/data settings
    p.add_argument('--approx', default='NRSur7dq4')
    p.add_argument('--use-higher-order-modes', action='store_true')

    p.add_argument('--sampling-rate', type=int, default=2048)

    p.add_argument('--flow', type=float, default=11,
                   help="lower frequency bound for data conditioning and likelihood function (ACF)")
    p.add_argument('--fmax', type=float, default=None,
                   help="Upper frequency bound for data conditioning and likelihood function (ACF) default: None")
    p.add_argument('--f22-start', type=float, default=11,
                   help="frequency at which to start generating 22 mode for waveforms, "
                        "NOTE! f22-start _is_ the reference frequency for eccentric waveforms ")
    p.add_argument('--fref', type=float, default=11)
    p.add_argument('--ifos', nargs='+', default=['H1', 'L1', 'V1'])

    # Optional args sampler settings
    p.add_argument('--nwalkers', type=int, default=512)
    p.add_argument('--nsteps', type=int, default=50000)
    p.add_argument('--ncpu', type=int, default=4)

    # Do we want to run with only the prior?
    p.add_argument('--only-prior', action='store_true')

    # Do we want to sample in time and/or sky position?
    p.add_argument('--vary-time', action='store_true')
    p.add_argument('--vary-skypos', action='store_true')
    p.add_argument('--vary-eccentricity', action='store_true')
    p.add_argument('--eccentricity-prior-bounds', type=float, nargs=2, default=[0, 0.9],
                   help="prior bounds for eccentricity parameter, only used if --vary-eccentricity given")
    p.add_argument('--total-mass-prior-bounds', type=float, nargs=2, default=[200, 350],
                   help="detector frame total mass bounds (in solar masses) for total_mass prior")
    p.add_argument('--mass-ratio-prior-bounds', type=float, nargs=2, default=[0.17, 1],
                   help="mass ratio bounds for mass ratio prior")
    p.add_argument('--luminosity-distance-prior-bounds', type=float, nargs=2, default=[100, 10000],
                   help="luminosity_distance bounds for luminosity_distance prior")
    p.add_argument('--spin-magnitude-prior-bounds', type=float, nargs=2, default=[0, 0.99],
                   help="Spin magnitude bounds for spin magnitude prior")
    p.add_argument('--time-prior-sigma', type=float, default=0.01, help="Standard deviation of time prior [s]")

    # Do we want to resume an old run?
    p.add_argument('--resume', action='store_true')

    # Debugging
    p.add_argument('--verbose', action='store_true')

    return p


def modify_parameters(data, args):
    """
    :param data:
    :param args:
    :return:
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, dict):
        df = pd.DataFrame([data])  # Convert the dictionary to a one-row DataFrame
    else:
        raise ValueError("Input must be either a DataFrame or a dictionary.")

    def equivocate_columns(dataframe, wanted_key, maybe_key):
        """
        Often parameters will have the same names,
        this function checks for and sets both names
        :param dataframe:
        :param wanted_key:
        :param maybe_key:
        :return:
        """
        if wanted_key not in dataframe.columns:
            if maybe_key in dataframe.columns:
                dataframe[wanted_key] = dataframe[maybe_key]
            else:
                print(f'warning! neither {wanted_key} nor {maybe_key} in df.columns!')

    # Common operations for both DataFrames and dictionaries
    equivocate_columns(df, 'geocenter_time', 'geocent_time')
    equivocate_columns(df, 'right_ascension', 'ra')
    equivocate_columns(df, 'declination', 'dec')
    equivocate_columns(df, 'mean_anomaly', 'mean_anomaly_periastron')
    equivocate_columns(df, 'polarization', 'psi')
    equivocate_columns(df, 'luminosity_distance', 'distance_mpc')

    if 'f_ref' not in df.columns:
        df['f_ref'] = args.fref

    if 'mass_1' not in df.columns and all(col in df.columns for col in ['total_mass', 'mass_ratio']):
        m1s, m2s = utils.m1m2_from_mtotq(df['total_mass'], df['mass_ratio'])
        df['mass_1'] = m1s
        df['mass_2'] = m2s

    if 'total_mass' not in df.columns:
        df['total_mass'] = df['mass_1'] + df['mass_2']

    if 'mass_ratio' not in df.columns:
        df['mass_ratio'] = df['mass_2'] / df['mass_1']

    # TODO not sure if these are the names all the time
    if 'eccentricity' not in df.columns:
        df['eccentricity'] = 0
    if 'mean_anomaly' not in df.columns:
        df['mean_anomaly'] = 0

    spin_component_keys = ['inclination', 'spin1_x', 'spin1_y', 'spin1_z', 'spin2_x', 'spin2_y', 'spin2_z']
    if not all(key in df.columns for key in spin_component_keys):
        # TODO do not overwrite existing parameters
        df[['inclination', 'spin1_x', 'spin1_y', 'spin1_z', 'spin2_x', 'spin2_y', 'spin2_z']] = df.apply(
            lambda row: pd.Series(utils.transform_spins(
                row['theta_jn'], row['phi_jl'],
                row['tilt_1'], row['tilt_2'],
                row['phi_12'], row['a_1'], row['a_2'],
                row['mass_1'], row['mass_2'],
                row['f_ref'], row['phase'])
            ), axis=1)
    else:
        # TODO do not overwrite existing parameters
        df[['theta_jn', 'phi_jl', 'tilt_1', 'tilt_2', 'phi_12', 'a_1', 'a_2']] = df.apply(
            lambda row: pd.Series(utils.transformPrecessingWvf2PE(
                row['inclination'],
                row['spin1_x'], row['spin1_y'], row['spin1_z'],
                row['spin2_x'], row['spin2_y'], row['spin2_z'],
                row['mass_1'], row['mass_2'],
                row['f_ref'], row['phase'])
            ), axis=1)

    if isinstance(data, pd.DataFrame):
        return df
    elif isinstance(data, dict):
        return df.to_dict(orient='records')[0]  # Convert back to dictionary


def get_injected_parameters(args, initial_run_dir='', verbose=False):
    """
    Loads in injection parameters from args.injected_paramters or, if there was no injection,
    loads in max likelihood samples from args.pe_posterior_h5_file
    :param args: argparser
    :param initial_run_dir:
    :param verbose: Level of detail printing output
    :return:
    """

    if (args.injected_parameters is None) and (args.reference_parameters is None) and (args.pe_posterior_h5_file is None):
        raise ValueError("WARNING: none of --injected-parameters, "
                         "--reference_parameters, or --pe-posterior-h5-file were given. "
                         " These parameters are needed in order to set up time cuts and initialize the sampler."
                         " Please provide one!")

    # if using real data
    if args.injected_parameters is None:
        # Use reference parameter from max logL in PE file...
        if args.reference_parameters is None:
            pe_posterior_h5_file = os.path.join(initial_run_dir, args.pe_posterior_h5_file)
            ref_pe_samples = utils.get_pe_samples(pe_posterior_h5_file)

            # "Injected parameters" = max(P) draw from the samples associated with this data
            log_prob = ref_pe_samples['log_likelihood'] + ref_pe_samples['log_prior']
            max_L_index = np.argmax(log_prob)
            reference_parameters = {field: ref_pe_samples[field][max_L_index] for field in ref_pe_samples.dtype.names}
        # set reference parameters to the passed in reference_parameters
        else:
            reference_parameters = utils.parse_injected_parameters(args.reference_parameters,
                                                                  initial_run_dir=initial_run_dir)

        if 'f_ref' not in reference_parameters.keys():
            reference_parameters['f_ref'] = args.fref
    # Else, generate an injection (currently, only set up for no noise case)
    else:
        # Load in injected parameters
        reference_parameters = utils.parse_injected_parameters(args.injected_parameters, initial_run_dir=initial_run_dir)

        # Check that the reference freqs line up
        err_msg = f"Injection fref={reference_parameters['f_ref']} does not equal sampler fref={args.fref}"
        assert reference_parameters['f_ref'] == args.fref, err_msg
        
        ref_pe_samples = None

    reference_parameters = modify_parameters(reference_parameters, args)

    if verbose:
        print('reference_parameters are', reference_parameters)
        
    return reference_parameters, ref_pe_samples


def initialize_kwargs(args, reference_parameters):
    # Check that a cutoff time is given
    assert args.Tcut_cycles is not None or args.Tcut_seconds is not None, "must give a cutoff time"

    # Check that the given mode is allowed
    run_mode = args.mode
    assert run_mode in ['full', 'pre', 'post'], f"mode must be 'full', 'pre', or 'post'. given mode = '{run_mode}'."

    """
    Arguments for the posterior function
    """

    # put all arguments into a dict
    kwargs = {
        'mtot_lim': args.total_mass_prior_bounds,
        'q_lim': args.mass_ratio_prior_bounds,
        'chi_lim': args.spin_magnitude_prior_bounds,
        'dist_lim': args.luminosity_distance_prior_bounds,
        'sigma_time': args.time_prior_sigma,
        'eccentricity_lim': args.eccentricity_prior_bounds,
        'approx': args.approx,
        'f_ref': args.fref,
        'f_low': args.flow,
        'f22_start': args.f22_start,
        'delta_t': 1 / args.sampling_rate,

        'right_ascension': reference_parameters['right_ascension'],
        'declination': reference_parameters['declination'],
        'polarization': reference_parameters['polarization'],
        'geocenter_time': reference_parameters['geocenter_time'],
        'verbose': args.verbose,
    }
    return kwargs


def make_waveform_manager(args, **kwargs):
    """
    :param args: argument parser args
    :param kwargs:
    :return:
    """
    if args.approx == 'TEOBResumSDALI':
        wf_manager = utils.NewWaveformManager(args.ifos,
                                              use_higher_order_modes=args.use_higher_order_modes,
                                              vary_time=args.vary_time,
                                              vary_skypos=args.vary_skypos,
                                              vary_eccentricity=args.vary_eccentricity, **kwargs)
    else:
        wf_manager = utils.WaveformManager(args.ifos,
                                           vary_time=args.vary_time,
                                           vary_skypos=args.vary_skypos,
                                           vary_eccentricity=args.vary_eccentricity,
                                           **kwargs)
    return wf_manager


def get_conditioned_time_and_data(args, wf_manager, reference_parameters, initial_run_dir='', verbose=False):
    # Check that a cutoff time is given
    assert args.Tcut_cycles is not None or args.Tcut_seconds is not None, "must give a cutoff time"

    # Check that the given mode is allowed
    run_mode = args.mode
    assert run_mode in ['full', 'pre', 'post'], f"mode must be 'full', 'pre', or 'post'. given mode = '{run_mode}'."

    # Unpack some basic parameters
    ifos = args.ifos
    data_path_dict, psd_path_dict = utils.parse_data_and_psds(args, initial_run_dir)

    """
    Load or generate data
    """

    # Either reads injection file or finds maxL parameters from
    tpeak_geocent = reference_parameters['geocenter_time']

    # PSDs
    pe_psds = {}
    for ifo in ifos:
        pe_psds[ifo] = np.genfromtxt(psd_path_dict[ifo], dtype=float)

    # If real data ...
    if args.injected_parameters is None:
        # Load data
        raw_time_dict, raw_data_dict = utils.load_raw_data(ifos=ifos, path_dict=data_path_dict, verbose=verbose)

    # Else, generate an injection (currently, only set up for no noise case)
    else:
        # Times
        raw_time_dict, _ = utils.load_raw_data(ifos=ifos, path_dict=data_path_dict, verbose=verbose)
        if verbose:
            print('reference_injection params: ', reference_parameters)
        raw_data_dict = wf_manager.get_projected_waveform(x_phys=reference_parameters, ifos=args.ifos,
                                                          time_dict=raw_time_dict,
                                                          f_ref=args.fref, f22_start=args.f22_start)

    # tcut = cutoff time in waveform
    if args.Tcut_seconds is not None:
        # option 1: truncation time given in seconds already
        tcut_geocent = tpeak_geocent + args.Tcut_seconds
    else:
        # option 2: find truncation time based off of # number of cycles from peak
        Ncycles = args.Tcut_cycles

        projected_waveform_dict = wf_manager.get_projected_waveform(reference_parameters,
                                                                    ifos=args.ifos,
                                                                    time_dict=raw_time_dict,
                                                                    f_ref=args.fref, f22_start=args.f22_start)

        tcut_geocent = utils.get_Tcut_from_Ncycles(projected_waveform_dict, raw_time_dict,
                                                   ifo="H1", Ncycles=Ncycles,
                                                   ra=reference_parameters['right_ascension'],
                                                   dec=reference_parameters['declination'])

    if verbose:
        print(f'\nCutoff time: {tcut_geocent}')
    tcut_dict, _ = utils.get_tgps_and_ap_dicts(tcut_geocent, ifos,
                                               reference_parameters['right_ascension'],
                                               reference_parameters['declination'],
                                               reference_parameters['polarization'])

    """
    Condition data
    """
    # icut = index corresponding to cutoff time
    time_dict, data_dict, icut_dict = utils.condition(raw_time_dict, raw_data_dict, tcut_dict,
                                                      args.sampling_rate, f_min=args.flow * 3 / 4,
                                                      f_max=args.fmax, verbose=verbose)

    # Time spacing of data
    dt = time_dict[ifos[0]][1] - time_dict[ifos[0]][0]

    # Decide how much data to analyze based of off run mode
    if run_mode == 'full':
        TPre = tcut_geocent - args.Tstart
        TPost = args.Tend - tcut_geocent
    elif run_mode == 'pre':
        TPre = tcut_geocent - args.Tstart
        TPost = 0
    elif run_mode == 'post':
        TPre = 0
        TPost = args.Tend - tcut_geocent
    else:
        raise NotImplementedError(f'Run mode {run_mode} is not defined, please use one of pre post or full')

    if TPre < 0:
        print(f"Warning! Seconds analyzed before cut is less than 0!: {TPre} from {args.Tstart} to {tcut_geocent}")
    if TPost < 0:
        print(f"Warning! Seconds analyzed after cut is less than 0!: {TPost} from {tcut_geocent} to {args.Tend}")

    # Duration --> number of time samples to look at
    # Note Npre cannot be greater than idx or else our times index from [-1: idx],
    # (so itll be empty), to avoid this off by 1 error we use np.floor
    Npre = int(np.floor(TPre / dt))
    Npost = int(
        round(TPost / dt)) + 1  # must add one so that the target time is actually included, even if Tpost = 0,
    # otherwise WF placement gets messed up
    Nanalyze = Npre + Npost
    Tanalyze = Nanalyze * dt
    if verbose:
        print('\nWill analyze {:.3f} s of data at {:.1f} Hz\n'.format(Tanalyze, 1 / dt))
    assert Tanalyze > 0, "Geocenter cut time must be between Tstart and Tend. Please Modify your run settings." \
                         f" Start to end: {args.Tstart}, {args.Tend} with cut at {tcut_geocent}"

    # Crop analysis data to specified duration.
    for ifo, idx in icut_dict.items():
        if Npre > idx:
            print("ERROR! You cannot have more points pre-cutoff time than there are points "
                    "between the start and the cutoff time")
        # idx = sample closest to desired time
        time_dict[ifo] = time_dict[ifo][idx - Npre:idx + Npost]
        data_dict[ifo] = data_dict[ifo][idx - Npre:idx + Npost]

    # Calculate ACF
    rho_dict, cond_psds = utils.get_ACF(pe_psds, time_dict, f_low=args.flow, f_max=args.fmax, return_psds=True)

    for ifo, rho in rho_dict.items():
        assert len(rho) == len(data_dict[ifo]), 'Length for ACF is not the same as for the data'

    return time_dict, data_dict, pe_psds


def main():

    verbose = True
    # Parse the commandline arguments
    p = create_run_sampler_arg_parser()
    args = p.parse_args()

    backend_path = args.output_h5  # where emcee spits its output

    reference_parameters, ref_pe_samples = get_injected_parameters(args, verbose=verbose)
    kwargs = initialize_kwargs(args, reference_parameters)

    wf_manager = make_waveform_manager(args, **kwargs)

    time_dict, data_dict, psd_dict = get_conditioned_time_and_data(args,
                                                                   wf_manager=wf_manager,
                                                                   reference_parameters=reference_parameters,
                                                                   verbose=verbose)

    if verbose:
        print(f"kwargs are:")
        print(kwargs)
        
    """
    Set up likelihood 
    """
    likelihood_manager = utils.LnLikelihoodManager(
        psd_dict=psd_dict, time_dict=time_dict,
        data_dict=data_dict,
        vary_time=args.vary_time, vary_skypos=args.vary_skypos,
        vary_eccentricity=args.vary_eccentricity,
        f_max=args.fmax,
        only_prior=args.only_prior,
        use_higher_order_modes=args.use_higher_order_modes,
        **kwargs)

    """
    Set up sampler
    """

    # Emcee args
    nsteps = args.nsteps
    nwalkers = args.nwalkers
    # Default num dimensions (fixed time and sky position) = 14
    ndim = likelihood_manager.num_parameters
    if verbose:
        print("Sampling %i parameters." % ndim)

    # Where to save samples while sampler running
    backend = emcee.backends.HDFBackend(backend_path)

    # Resume if we want
    if args.resume and os.path.isfile(backend_path):

        # Load in last sample to use as the new starting walkers
        try:
            p0 = backend.get_last_sample()
            # adjust number of steps
            nsteps_already_taken = backend.get_chain().shape[0]
            nsteps = nsteps - nsteps_already_taken
        except AttributeError:
            # This happens when the file has been made but the job crashed before writing any points to it.
            # So we overwrite it
            # Reset the backend
            print('WARNING! Backend was empty, resetting backend')
            try:
                backend.reset(nwalkers, ndim)
            except OSError:
                print("resetting backend did not seem to fix issue, deleting h5 file and making new backend")
                os.remove(backend_path)
                backend = emcee.backends.HDFBackend(backend_path)
                backend.reset(nwalkers, ndim)

            p0 = likelihood_manager.log_prior.initialize_walkers(
                nwalkers, reference_parameters, reference_posteriors=ref_pe_samples, verbose=verbose
            )

    else:
        # Reset the backend
        backend.reset(nwalkers, ndim)
        p0 = likelihood_manager.log_prior.initialize_walkers(
            nwalkers, reference_parameters, reference_posteriors=ref_pe_samples, verbose=verbose
        )

    # Deactivate numpy default number of cores to avoid using too many
    if args.ncpu > 1:
        os.environ["OMP_NUM_THREADS"] = "1"

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(nsteps)

    # This will be useful for testing convergence
    old_tau = np.inf

    # now use multiprocessing
    def sort_on_runtime(pos):
        p = np.atleast_2d(pos)
        idx = np.argsort(p[:, 0])[::-1]
        return p[idx], idx

    """
    Run sampler
    """

    print("Running with %i cores." % args.ncpu)
    with closing(Pool(processes=args.ncpu)) as pool:

        sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood_manager.get_lnprob,
                                        backend=backend, pool=pool,
                                        runtime_sortingfn=sort_on_runtime,
                                        kwargs=kwargs)

        # If there are still iterations left to run ... 
        if nsteps > 0:

            # Cycle through remaining iterations
            for sample in sampler.sample(p0, iterations=nsteps, progress=True):

                # Only check convergence every 100 steps
                if sampler.iteration % 100:
                    continue

                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even
                # if it isn't trustworthy
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1

                # Check convergence
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break  # if convergence reached before nsteps, stop running
                old_tau = tau

        pool.terminate()

    """
    Post processing and saving data
    """
    # Print dimensions of chain
    print(sampler.get_chain().shape)

    # Postprocessing
    df = utils.postprocess_samples(sampler,
                                   likelihood_manager.log_prior,
                                   **kwargs)

    # Save
    sample_path = backend_path.replace('h5', 'dat')
    df.to_csv(sample_path, sep=' ', index=False)
    print("File saved: %r" % sample_path)


if __name__ == "__main__":
    main()
