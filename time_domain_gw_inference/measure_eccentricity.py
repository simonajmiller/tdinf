import argparse
import pandas as pd
import os
from time_domain_gw_inference import group_postprocess
import run_sampler
from tqdm import tqdm
import astropy.units as u
import astropy.constants as constants
import numpy as np
import multiprocessing as mp
from itertools import repeat
try:
    import EOBRun_module
    from gwsignal.models.teobresums import convert_parameters_to_teob, TEOB_DALI_MODES_FROM_K
    from gw_eccentricity import measure_eccentricity
except ImportError:
    print("Warning! Unable to import EOBRun_module, will not be able to calculate eccentricities ")

def EOBRunPy(generator, **parameters):
    #print(parameters)
    # remove fixed and model-dependent pars from parameters
    #fixed_pars = {}
    generator.parameter_check(units_sys="Cosmo",
                              extra_parameters=generator.metadata['extra_parameters'],
                              **parameters)
    generator.waveform_dict = generator._strip_units(generator.waveform_dict)
    #generator.waveform_dict.update(**fixed_pars)
    teob_parameters = convert_parameters_to_teob(generator.waveform_dict)
    teob_parameters['anomaly'] = teob_parameters['anomaly'][0] # for some reason puts it in a list

    teob_parameters["use_mode_lm"] = generator._available_modes_teob_convention
    t, hp, hc, htlm, dyn = EOBRun_module.EOBRunPy(teob_parameters)
    return t, hp, hc, htlm, dyn


def get_actual_eccentricity(parameters, waveform_manager, delta_t, f_ref, f22_start, fref_in=None, tref_in=None,
                            method='Amplitude', num_orbits_to_exclude_before_merger=2, **kwargs):
    generator = waveform_manager.generator

    wf_dict = waveform_manager.physical_dict_to_waveform_dict(parameters)
    wf_dict['f22_start'] = f22_start * u.Hz
    wf_dict['deltaT'] = delta_t * u.s
    wf_dict['f22_ref'] = f_ref * u.Hz
    wf_dict['condition'] = 0

    t, hp, hc, htlm, dyn = EOBRunPy(generator, **wf_dict)
    modeDict = {TEOB_DALI_MODES_FROM_K[int(ind)]: mode_wf[0] * np.exp(-1j * mode_wf[1])
                for ind, mode_wf in htlm.items()}

    dt_M = (constants.G / constants.c ** 3 * u.Msun * parameters['total_mass']).to(u.s)

    if tref_in is not None:
        t = t / dt_M.value
        tref_in = tref_in
    dataDict = {'t': t, 'hlm': modeDict}

    try:
        res = measure_eccentricity(fref_in=fref_in, tref_in=tref_in,
                                   dataDict=dataDict,
                                   method=method,
                                   num_orbits_to_exclude_before_merger=num_orbits_to_exclude_before_merger,
                                   **kwargs)
        # For debugging
        # import matplotlib.pyplot as plt
        # gwecc_object = res["gwecc_object"]
        # fig, ax = gwecc_object.make_diagnostic_plots()
        # plt.savefig('diagnostic.png')
        # plt.show()

    except Exception as e:
        print(f'measure eccentricity failed with error {e} \n \t from parameters {parameters}')
        res = {'eccentricity': np.nan, 'mean_anomaly': np.nan}
        return res
    return res


def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, tqdm(args_for_starmap, total=len(args_iter)))

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


def calculate_eccentricity(row, lm, delta_t, fref_in=None, tref_in=None, **kwargs):
    res = get_actual_eccentricity(row.to_dict(), lm.waveform_manager, delta_t,
                                  f_ref=lm.f_ref,
                                  f22_start=lm.f22_start,
                                  fref_in=fref_in, tref_in=tref_in, **kwargs)
    if tref_in is None:
        ecc_key = f'eccentricity_{fref_in}'
        mean_anomaly_key = f'mean_anomaly_{fref_in}'
    else:
        ecc_key = f'eccentricity_tref_{tref_in}'
        mean_anomaly_key = f'mean_anomaly_tref_{tref_in}'

    row[ecc_key] = res['eccentricity']
    row[mean_anomaly_key] = res['mean_anomaly']
    return row


# def get_new_dataframe_name(run_directory, run_key):
#     """ pass in directory where runs are stored, not the waveform directory"""
#     return os.path.join(waveform_dir, f'{run_key}_waveforms.h5')


#def __main__():
if __name__ == "__main__":
    print('starting!')

    parser = argparse.ArgumentParser()
    parser.add_argument('--fref_in', type=float, default=None,
                        help="frequency hz to measure eccentricity from, default None"
                             "One of fref_in or tref_in must be supplied ")
    parser.add_argument('--tref_in', type=float, default=None,
                        help="reference time in (dimensionless, total mass units M) to measure eccentricity from, default None"
                             "One of fref_in or tref_in must be supplied ")

    # Add arguments
    parser.add_argument("--directory", type=str, help="Input directory")
    parser.add_argument("--run_key", type=str, help="name of run for which to create waveforms")
    parser.add_argument("--overwrite", action="store_true",
                        help="Flag to overwrite existing files (default: False)")
    parser.add_argument("--append", action="store_true",
                        help="Flag to compute additional eccentricity parameters if file already exists")

    parser.add_argument("--ncpu", type=int, default=mp.cpu_count(), help=f"Number of parallel processes to start,"
                                                                         f" default {mp.cpu_count()}")
    parser.add_argument("--debug_level", type=int, default=0, help="level of verbosity for measure eccentricity\n"
                                                                   "\t-1: All warnings are suppressed. NOTE: Use at your own risk! \n"
                                                                   "\t0: Only important warnings are issued. \n"
                                                                   "\t1: All warnings are issued. Use when investigating. \n"
                                                                   "\t2: All warnings become exceptions.\n")

    parser.add_argument('--method', type=str, default="AmplitudeFits",
                        help="""
        Which waveform data to use for finding extrema. Options are:
        - "Amplitude": Finds extrema of Amp22(t).
        - "Frequency": Finds extrema of omega22(t).
        - "ResidualAmplitude": Finds extrema of resAmp22(t), the residual
          amplitude, obtained by subtracting the Amp22(t) of the quasicircular
          counterpart from the Amp22(t) of the eccentric waveform. The
          quasicircular counterpart is described in the documentation of
          dataDict below.
        - "ResidualFrequency": Finds extrema of resomega22(t), the residual
          frequency, obtained by subtracting the omega22(t) of the
          quasicircular counterpart from the omega22(t) of the eccentric
          waveform.
        - "AmplitudeFits": Uses Amp22(t) and iteratively subtracts a
          PN-inspired fit of the extrema of Amp22(t) from it, and finds extrema
          of the residual.
        - "FrequencyFits": Uses omega22(t) and iteratively subtracts a
          PN-inspired fit of the extrema of omega22(t) from it, and finds
          extrema of the residual.
    
        The available list of methods can be also obtained from
        gw_eccentricity.get_available_methods().
        Detailed description of these methods can be found in Sec. III of
        arXiv:2302.11257.
    
        The Amplitude and Frequency methods can struggle for very small
        eccentricities, especially near the merger, as the
        secular amplitude/frequency growth dominates the modulations due to
        eccentricity, making extrema finding difficult.
    
        The ResidualAmplitude/ResidualFrequency/AmplitudeFits/FrequencyFits
        methods avoid this limitation by removing the secular growth before
        finding extrema. However, methods that use the frequency for finding
        extrema (Frequency/ResidualFrequency/FrequencyFits) can be more
        sensitive to junk radiation in NR data.
    
        Therefore, the recommended methods are
        ResidualAmplitude/AmplitudeFits/Amplitude""")

    args = parser.parse_args()

    tref_in = args.tref_in
    fref_in = args.fref_in

    # Assign arguments to variables
    directory = args.directory
    overwrite = args.overwrite

    # don't overwrite existing files unless asked to explicitly
    filename_dict = group_postprocess.generate_filename_dict(directory)
    dataframe_file = os.path.join(filename_dict[args.run_key], os.path.basename(filename_dict[args.run_key]) + '.dat')

    dirname = os.path.dirname(dataframe_file)
    new_sample_filename = 'new_' + os.path.basename(dataframe_file)
    new_sample_path = os.path.join(dirname, new_sample_filename)

    if not overwrite and (not args.append) and os.path.exists(new_sample_path):

        # print('loading from existing file')
        print('new calculated dataframe already exists, use --overwrite in order to overwrite all the files,'
              ' or --append to calculate new columns')
        exit()

    run_parser = run_sampler.create_run_sampler_arg_parser()
    run_args, \
    run_kwargs, \
    run_likelihood_manager = group_postprocess.get_settings_from_command_line_file(
        os.path.join(directory, 'command_line.sh'),
        filename_dict[args.run_key],
        directory + '/',
        run_parser, verbose=True)

    sub_directory = filename_dict[args.run_key]
    dataframe = group_postprocess.load_dataframe(directory, sub_directory)

    if args.append and os.path.exists(new_sample_path):
        new_df = group_postprocess.load_dataframe_and_parameters(new_sample_path)
    else:
        new_df = dataframe.copy()

    new_df = new_df.copy() # just makes dataframe shorter for tests
    if tref_in is not None:
        eccentricity_key = f'eccentricity_tref_{tref_in}'
        mean_anomaly_key = f'mean_anomaly_tref_{tref_in}'
    elif fref_in is not None:
        eccentricity_key = f'eccentricity_{fref_in}'
        mean_anomaly_key = f'mean_anomaly_{fref_in}'
    else:
        raise RuntimeError(f"One of tref_in or fref_in must not be none!")

    if {eccentricity_key, mean_anomaly_key}.issubset(new_df.columns) and args.append and (not args.overwrite):
        print(f'WARNING: new calculated dataframe already exists WITH {eccentricity_key} and {mean_anomaly_key}'
              ' but! you have run with --append. \n'
              'Use --overwrite in order to overwrite this dataframe entirely, '
              'or else sit happy knowing your calculations have already been done')
        exit()
    else:
        new_df[eccentricity_key] = 0
        new_df[mean_anomaly_key] = 0

    ifo = run_likelihood_manager.ifos[0]
    delta_t = run_likelihood_manager.time_dict[ifo][1] - run_likelihood_manager.time_dict[ifo][0]
    args_iter = [(new_df.iloc[i], run_likelihood_manager, delta_t) for i in range(len(new_df))]

    kwargs_iter = repeat(dict(fref_in=fref_in, method=args.method, tref_in=tref_in,
                              extra_kwargs=dict(debug_level=args.debug_level)))

    with mp.Pool(processes=args.ncpu) as pool:
        # Use pool.starmap to parallelize the computation
        results = starmap_with_kwargs(pool, calculate_eccentricity, args_iter, kwargs_iter)

    results_df = pd.DataFrame.from_records(results)

    print('new df is', results_df)
    failed = results_df.isnull().any(axis=1).sum()
    print(f'failed is {failed} of {len(results_df)}, {100 * failed / len(results_df)}%')
    results_df.to_csv(new_sample_path, sep=' ', index=False, na_rep='nan')
    print('All done!')







