#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import emcee
import scipy.linalg as sl
import pandas as pd
from multiprocessing import Pool
from contextlib import closing
import os
import sys
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

    # Optional args for waveform/data settings
    p.add_argument('--approx', default='NRSur7dq4')
    p.add_argument('--downsample', type=int, default=8)
    p.add_argument('--flow', type=float, default=11)
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

    # Do we want to resume an old run?
    p.add_argument('--resume', action='store_true')

    return p


def initialize_kwargs(args):
    # Check that a cutoff time is given
    assert args.Tcut_cycles is not None or args.Tcut_seconds is not None, "must give a cutoff time"

    # Check that the given mode is allowed
    run_mode = args.mode
    assert run_mode in ['full', 'pre', 'post'], f"mode must be 'full', 'pre', or 'post'. given mode = '{run_mode}'."

    # Unpack some basic parameters
    ifos = args.ifos
    data_path_dict, psd_path_dict = utils.parse_data_and_psds(args)
    pe_posterior_h5_file = args.pe_posterior_h5_file
    backend_path = args.output_h5  # where emcee spits its output
    f_ref = args.fref
    f_low = args.flow
    ds_factor = args.downsample

    print('')

    """
    Load or generate data
    """

    # If real data ...
    if args.injected_parameters is None:

        # Load data
        raw_time_dict, raw_data_dict = utils.load_raw_data(ifos=ifos, path_dict=data_path_dict)
        pe_out = utils.get_pe(raw_time_dict, pe_posterior_h5_file, verbose=False, psd_path_dict=psd_path_dict,
                              f_ref=f_ref, f_low=f_low)
        tpeak_geocent, pe_samples, log_prob, pe_psds, skypos = pe_out

        # "Injected parameters" = max(P) draw from the samples associated with this data
        injected_parameters = pe_samples[np.argmax(log_prob)]

        ## tpeak = placement of waveform
        print('\nWaveform placement time:')
        tpeak_dict, ap_dict = utils.get_tgps_and_ap_dicts(tpeak_geocent, ifos, skypos['ra'], skypos['dec'],
                                                          skypos['psi'])

    # Else, generate an injection (currently, only set up for no noise case)
    else:

        # Load in injected parameters
        injected_parameters = utils.parse_injected_parameters(args.injected_parameters)

        # Check that the reference freqs line up
        err_msg = f"Injection fref={injected_parameters['f_ref']} does not equal sampler fref={f_ref}"
        assert injected_parameters['f_ref'] == f_ref, err_msg

        # Triggertime and sky position
        tpeak_geocent = injected_parameters['geocent_time']
        skypos = {k: injected_parameters[k] for k in ['ra', 'dec', 'psi']}

        ## tpeak = placement of waveform
        print('\nWaveform placement time:')
        tpeak_dict, ap_dict = utils.get_tgps_and_ap_dicts(tpeak_geocent, ifos, skypos['ra'], skypos['dec'],
                                                          skypos['psi'])

        # PSDs
        pe_psds = {}
        for ifo in ifos:
            pe_psds[ifo] = np.genfromtxt(psd_path_dict[ifo], dtype=float)

        # Times
        raw_time_dict = utils.load_raw_data(ifos=ifos, path_dict=data_path_dict)[0]

        # Injection
        raw_data_dict = utils.injectWaveform(parameters=injected_parameters, time_dict=raw_time_dict,
                                             tpeak_dict=tpeak_dict, ap_dict=ap_dict, skypos=skypos,
                                             f_ref=f_ref, f_low=f_low, approx=args.approx)

    ## tcut = cutoff time in waveform
    if args.Tcut_seconds is not None:
        # option 1: truncation time given in seconds already
        tcut_geocent = tpeak_geocent + args.Tcut_seconds
    else:
        # option 2: find truncation time based off of # number of cycles from peak
        Ncycles = args.Tcut_cycles
        tcut_geocent = utils.get_Tcut_from_Ncycles(Ncycles, parameters=injected_parameters, time_dict=raw_time_dict,
                                                   tpeak_dict=tpeak_dict, ap_dict=ap_dict, skypos=skypos, f_ref=f_ref,
                                                   f_low=f_low, approx=args.approx)

    print('\nCutoff time:')
    tcut_dict, _ = utils.get_tgps_and_ap_dicts(tcut_geocent, ifos, skypos['ra'], skypos['dec'], skypos['psi'])

    # If we are varying skyposition
    if args.vary_skypos:
        ap_dict = None  # don't want fixed antenna patterns

    # If we are varying over time of coalescence
    if args.vary_time:
        tpeak_dict = None  # don't want fixed time of arrival at detectors

    """
    Condition data
    """

    # icut = index corresponding to cutoff time
    time_dict, data_dict, icut_dict = utils.condition(raw_time_dict, raw_data_dict, tcut_dict, ds_factor, f_low)

    # Time spacing of data
    dt = time_dict['H1'][1] - time_dict['H1'][0]

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

    # Duration --> number of time samples to look at
    Npre = int(round(TPre / dt))
    Npost = int(round(TPost / dt)) + 1  # must add one so that the target time is actually included, even if Tpost = 0,
    # otherwise WF placement gets messed up
    Nanalyze = Npre + Npost
    Tanalyze = Nanalyze * dt
    print('\nWill analyze {:.3f} s of data at {:.1f} Hz\n'.format(Tanalyze, 1 / dt))

    # Crop analysis data to specified duration.
    for ifo, idx in icut_dict.items():
        # idx = sample closest to desired time
        time_dict[ifo] = time_dict[ifo][idx - Npre:idx + Npost]
        data_dict[ifo] = data_dict[ifo][idx - Npre:idx + Npost]

    # Calculate ACF
    rho_dict = utils.get_ACF(pe_psds, time_dict, f_low=f_low)

    for ifo, rho in rho_dict.items():
        assert len(rho) == len(data_dict[ifo]), 'Length for ACF is not the same as for the data'

    """
    Arguments for the posterior function
    """

    # configure mass prior
    inj_mtot = injected_parameters['mass_1'] + injected_parameters['mass_2']
    max_mass_prior = np.ceil(inj_mtot + 100)
    min_mass_prior = np.maximum(np.floor(inj_mtot - 100), 5)

    # configure distance prior
    inj_dist_log = np.log10(injected_parameters['luminosity_distance'])
    min_dist_prior = int(np.power(10, np.floor(inj_dist_log - 1)))
    max_dist_prior = min(10000, int(np.power(10, np.ceil(inj_dist_log + 1))))  # cap max distance at 10000 MPc

    # put all arguments into a dict
    kwargs = {
        'mtot_lim': [min_mass_prior, max_mass_prior],
        'q_lim': [0.17, 1],
        'chi_lim': [0, 0.99],
        'dist_lim': [min_dist_prior, max_dist_prior],

        'approx': args.approx,
        'f_ref': f_ref,
        'f_low': f_low,
        'only_prior': args.only_prior,
        'delta_t': dt,

        'ra': skypos['ra'],  # default right ascension if not varied
        'dec': skypos['dec'],  # default declination if not varied
        'psi': skypos['psi'],  # default polarization if not varied
        'tgps_geocent': tpeak_geocent,  # default waveform placement time if not varied

        'rho_dict': rho_dict,
        'time_dict': time_dict,
        'data_dict': data_dict,
        'ap_dict': ap_dict,
        'tpeak_dict': tpeak_dict
    }
    return kwargs


def main():
    
    # Parse the commandline arguments
    p = create_run_sampler_arg_parser()
    args = p.parse_args()
    
    # Check that a cutoff time is given
    assert args.Tcut_cycles is not None or args.Tcut_seconds is not None, "must give a cutoff time"

    # Check that the given mode is allowed
    run_mode = args.mode
    assert run_mode in ['full', 'pre', 'post'], f"mode must be 'full', 'pre', or 'post'. given mode = '{run_mode}'."

    # Unpack some basic parameters
    ifos = args.ifos
    data_path_dict, psd_path_dict = utils.parse_data_and_psds(args)
    pe_posterior_h5_file = args.pe_posterior_h5_file
    backend_path = args.output_h5  # where emcee spits its output
    f_ref = args.fref
    f_low = args.flow
    ds_factor = args.downsample

    print('')

    """
    Load or generate data
    """

    # If real data ...
    if args.injected_parameters is None:

        # Load data
        raw_time_dict, raw_data_dict = utils.load_raw_data(ifos=ifos, path_dict=data_path_dict)
        pe_out = utils.get_pe(raw_time_dict, pe_posterior_h5_file, verbose=False, psd_path_dict=psd_path_dict, 
                             f_ref=f_ref, f_low=f_low)
        tpeak_geocent, pe_samples, log_prob, pe_psds, skypos = pe_out

        # "Injected parameters" = max(P) draw from the samples associated with this data
        injected_parameters = pe_samples[np.argmax(log_prob)]

        ## tpeak = placement of waveform
        print('\nWaveform placement time:')
        tpeak_dict, ap_dict = utils.get_tgps_and_ap_dicts(tpeak_geocent, ifos, skypos['ra'], skypos['dec'],
                                                          skypos['psi'])

    # Else, generate an injection (currently, only set up for no noise case)
    else:

        # Load in injected parameters
        injected_parameters = utils.parse_injected_parameters(args.injected_parameters)

        # Check that the reference freqs line up
        err_msg = f"Injection fref={injected_parameters['f_ref']} does not equal sampler fref={f_ref}"
        assert injected_parameters['f_ref'] == f_ref, err_msg

        # Triggertime and sky position
        tpeak_geocent = injected_parameters['geocent_time']
        skypos = {k: injected_parameters[k] for k in ['ra', 'dec', 'psi']}

        ## tpeak = placement of waveform
        print('\nWaveform placement time:')
        tpeak_dict, ap_dict = utils.get_tgps_and_ap_dicts(tpeak_geocent, ifos, skypos['ra'], skypos['dec'],
                                                          skypos['psi'])

        # PSDs
        pe_psds = {}
        for ifo in ifos:
            pe_psds[ifo] = np.genfromtxt(psd_path_dict[ifo], dtype=float)

        # Times
        raw_time_dict = utils.load_raw_data(ifos=ifos, path_dict=data_path_dict)[0]

        # Injection
        raw_data_dict = utils.injectWaveform(parameters=injected_parameters, time_dict=raw_time_dict,
                                             tpeak_dict=tpeak_dict, ap_dict=ap_dict, skypos=skypos,
                                             f_ref=f_ref, f_low=f_low, approx=args.approx)

    ## tcut = cutoff time in waveform
    if args.Tcut_seconds is not None: 
        # option 1: truncation time given in seconds already
        tcut_geocent = tpeak_geocent + args.Tcut_seconds
    else: 
        # option 2: find truncation time based off of # number of cycles from peak
        Ncycles = args.Tcut_cycles 
        tcut_geocent = utils.get_Tcut_from_Ncycles(Ncycles, parameters=injected_parameters, time_dict=raw_time_dict,
                                                       tpeak_dict=tpeak_dict, ap_dict=ap_dict, skypos=skypos, f_ref=f_ref,
                                                       f_low=f_low, approx=args.approx) 

    print('\nCutoff time:')
    tcut_dict, _ = utils.get_tgps_and_ap_dicts(tcut_geocent, ifos, skypos['ra'], skypos['dec'], skypos['psi'])

    # If we are varying skyposition
    if args.vary_skypos:
        ap_dict = None  # don't want fixed antenna patterns

    # If we are varying over time of coalescence
    if args.vary_time:
        tpeak_dict = None  # don't want fixed time of arrival at detectors

    """
    Condition data
    """

    # icut = index corresponding to cutoff time
    time_dict, data_dict, icut_dict = utils.condition(raw_time_dict, raw_data_dict, tcut_dict, ds_factor, f_low)

    # Time spacing of data
    dt = time_dict['H1'][1] - time_dict['H1'][0]

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

    # Duration --> number of time samples to look at
    Npre = int(round(TPre / dt))
    Npost = int(round(TPost / dt)) + 1  # must add one so that the target time is actually included, even if Tpost = 0,
                                        # otherwise WF placement gets messed up
    Nanalyze = Npre + Npost
    Tanalyze = Nanalyze * dt
    print('\nWill analyze {:.3f} s of data at {:.1f} Hz\n'.format(Tanalyze, 1 / dt))

    # Crop analysis data to specified duration.
    for ifo, idx in icut_dict.items():
        # idx = sample closest to desired time
        time_dict[ifo] = time_dict[ifo][idx - Npre:idx + Npost]
        data_dict[ifo] = data_dict[ifo][idx - Npre:idx + Npost]

    # Calculate ACF
    rho_dict = utils.get_ACF(pe_psds, time_dict, f_low=f_low)
    
    for ifo,rho in rho_dict.items(): 
        assert len(rho) == len(data_dict[ifo]), 'Length for ACF is not the same as for the data'

    """
    Arguments for the posterior function
    """

    # configure mass prior
    inj_mtot = injected_parameters['mass_1'] + injected_parameters['mass_2']
    max_mass_prior = np.ceil(inj_mtot + 100)
    min_mass_prior = np.maximum(np.floor(inj_mtot - 100), 5)
    
    # configure distance prior
    inj_dist_log = np.log10(injected_parameters['luminosity_distance'])
    min_dist_prior = int(np.power(10, np.floor(inj_dist_log-1)))
    max_dist_prior = min(10000, int(np.power(10, np.ceil(inj_dist_log+1)))) # cap max distance at 10000 MPc

    # put all arguments into a dict
    kwargs = {
        'mtot_lim': [min_mass_prior, max_mass_prior],
        'q_lim': [0.17, 1],
        'chi_lim': [0, 0.99],
        'dist_lim': [min_dist_prior, max_dist_prior],

        'approx': args.approx,
        'f_ref': f_ref,
        'f_low': f_low,
        'only_prior': args.only_prior,
        'delta_t': dt,

        'ra': skypos['ra'],  # default right ascension if not varied
        'dec': skypos['dec'],  # default declination if not varied
        'psi': skypos['psi'],  # default polarization if not varied
        'tgps_geocent': tpeak_geocent,  # default waveform placement time if not varied

        'rho_dict': rho_dict,
        'time_dict': time_dict,
        'data_dict': data_dict,
        'ap_dict': ap_dict,
        'tpeak_dict': tpeak_dict
    }

    print(f"kwargs are:")
    print(kwargs)
    """
    Set up likelihood 
    """
    likelihood_manager = utils.LnLikelihoodManager(
        vary_time=args.vary_time, vary_skypos=args.vary_skypos, **kwargs)

    """
    Set up sampler
    """

    # Emcee args
    nsteps = args.nsteps
    nwalkers = args.nwalkers
    # Default num dimensions (fixed time and sky position) = 14
    ndim = 14
    # If we want to vary time ...
    if args.vary_time:
        ndim += 1
    # If we want to vary sky position ...
    if args.vary_skypos:
        ndim += 5  # add ra_x, ra_y, sin_dec, psi_x, psi_y

    print("Sampling %i parameters." % ndim)

    # Where to save samples while sampler running
    backend = emcee.backends.HDFBackend(backend_path)

    # Resume if we want
    if args.resume and os.path.isfile(backend_path):
        
        # Load in last sample to use as the new starting walkers
        p0 = backend.get_last_sample()
        
        # adjust number of steps
        nsteps_already_taken = backend.get_chain().shape[0]
        nsteps = nsteps - nsteps_already_taken
        
    else:
        # Reset the backend
        backend.reset(nwalkers, ndim)

        # Initialize walkers
        # (code sees unit scale quantities; use logit transformations
        # to take boundaries to +/- infinity)
        p0_arr = np.asarray([[np.random.normal() for j in range(ndim)] for i in range(nwalkers)])

        print(p0_arr)
        
        # replace some parameters (masses, spin mag, distance) in tight balls around their injected values
        for p, param_kw, lim_kw in zip(range(5), ['mtot', 'q', 'a_1', 'a_2', 'luminosity_distance'], 
                                    ['mtot_lim', 'q_lim', 'chi_lim', 'chi_lim', 'dist_lim']):
            # get physical parameter
            if param_kw=='mtot': 
                param_phys = injected_parameters['mass_1'] + injected_parameters['mass_2']
            elif param_kw=='q': 
                param_phys = injected_parameters['mass_2'] / injected_parameters['mass_1']
            else:
                param_phys = injected_parameters[param_kw]
            
            # transform into logistic space
            param_logit = utils.logit(param_phys, * kwargs[lim_kw])
            
            # draw gaussian ball in logistic space
            p0_arr[:,p] = np.asarray([np.random.normal(loc=param_logit, scale=0.05) for i in range(nwalkers)])

        # if time of coalescence sampled over need to include this separately since it isn't a unit scaled quantity
        if args.vary_time:
            dt_1M = 0.00127
            sigma_time = dt_1M * 2.5  # time prior from LVK has width of ~2.5M
            initial_t_walkers = np.random.normal(loc=tpeak_geocent, scale=sigma_time, size=nwalkers)
            p0_arr[:, ndim - 1] = initial_t_walkers  # time always saved as the final param

        p0 = p0_arr.tolist()

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
