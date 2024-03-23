from . import run_sampler
from . import utils
import os
import pandas as pd
import re
from matplotlib.lines import Line2D
import imageio

def load_run_settings_from_directory(directory, filename_dict=None):
    """
    Given a directory with runs created via the pipeline, load in args, kwargs, likelihood_manager, and dataframe
    :param directory: directory containing runs
    :param filename_dict: default None, dictionary describing the way that runs will be organized
    :return:
    dict {
     'dir': directory,
     'runs': {'full': {'likelihood_manager':..., 'args':, 'kwargs':},
     }
    }
    """
    if filename_dict is None:
        filename_dict = generate_filename_dict(directory)

    settings = {'dir': directory, 'runs': {key: {'args': None,
                                                 'kwargs': None,
                                                 'likelihood_manager': None} for key in filename_dict.keys()},
                'dfs': {key: None for key in filename_dict.keys()}}

    parser = run_sampler.create_run_sampler_arg_parser()

    for run in settings.keys():
        td_settings = settings['runs']
        td_samples = settings['dfs']
        directory = settings['dir']
        for key in td_settings.keys():
            print(f'key: {key}')
            try:
                td_settings[key]['args'], \
                td_settings[key]['kwargs'], \
                td_settings[key]['likelihood_manager'] = get_settings_from_command_line_file(
                    os.path.join(directory, 'command_line.sh'),
                    filename_dict[key],
                    directory + '/',
                    parser, verbose=True)
            except TypeError:
                print(f'unable to make {run} {key}')

            filename = os.path.join(directory, filename_dict[key] + '.dat')
            if not os.path.exists(filename):
                filename = os.path.join(directory, filename_dict[key] + '/' + filename_dict[key] + '.dat')

            try:
                td_samples[key] = pd.read_csv(filename, delimiter='\s+')
            except:
                continue
            td_samples[key] = calc_additional_parameters(td_samples[key])
    return settings


def get_settings_from_command_line_string(command_line_string, initial_run_dir, parser, verbose=False):
    skip_initial_arg = command_line_string.split()[1:]
    args = parser.parse_args(skip_initial_arg)
    reference_parameters = run_sampler.get_injected_parameters(args, initial_run_dir)
    kwargs = run_sampler.initialize_kwargs(args, reference_parameters)
    print('making wf manager')
    wf_manager = utils.NewWaveformManager(args.ifos,
                                          vary_time=args.vary_time,
                                          vary_skypos=args.vary_skypos,
                                          vary_eccentricity=args.vary_eccentricity, **kwargs)
    print('getting conditioned time and data')
    time_dict, data_dict, psd_dict = run_sampler.get_conditioned_time_and_data(args,
                                                                               wf_manager=wf_manager,
                                                                               reference_parameters=reference_parameters,
                                                                               initial_run_dir=initial_run_dir)

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
        only_prior=args.only_prior,
        **kwargs)
    return args, kwargs, likelihood_manager


def get_settings_from_command_line_file(command_line_file, file_prefix, initial_run_dir, parser, **kwargs):
    with open(command_line_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if file_prefix in line:
                print("line is", line)
                return get_settings_from_command_line_string(line, initial_run_dir, parser, **kwargs)


def calc_additional_parameters(df):
    # Calculate component masses
    m1s, m2s = utils.m1m2_from_mtotq(df['total_mass'], df['mass_ratio'])

    # Calculate chi-eff
    chieffs = utils.chi_effective(m1s, df['spin1_magnitude'], df['tilt1'], m2s, df['spin2_magnitude'], df['tilt2'])

    # Calculate chi-p
    chips = utils.chi_precessing(m1s, df['spin1_magnitude'], df['tilt1'], m2s, df['spin2_magnitude'], df['tilt2'])

    df['mass1'] = m1s
    df['mass2'] = m2s

    df['chi_effective'] = chieffs
    df['chi_precessing'] = chips

    return df


def generate_filename_dict(directory):
    filename_dict = {}
    pattern = re.compile(r'(post|pre|full)_(\d+\.\d+)seconds')

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            prefix, nseconds = match.groups()
            if prefix == 'full':
                filename_dict[prefix] = filename
            else:
                filename_dict[f"{prefix}_{nseconds}"] = filename

    return filename_dict


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy


def whiten_waveform_dict(wf_dict, choelsky_rho_dict):
    whitened_wf_dict = {}
    for ifo in wf_dict.keys():
        whitened_wf_dict[ifo] = np.linalg.solve(choelsky_rho_dict[ifo], wf_dict[ifo])
    return whitened_wf_dict


def get_choelsky_decomp_of_toeplitz(toeplitz_matrix):
    # compute covariance matrix  C and its Cholesky decomposition L (~sqrt of C)
    C = scipy.linalg.toeplitz(toeplitz_matrix)
    L = np.linalg.cholesky(C)
    return L


def make_gif(tc_floats, wfs_at_tc_list, reference_waveform_dict, dfs_at_tc, full_df,
             full_likelihood_manager,
             plot_param, prior_df = None, 
             ifo='L1', save_dir = None, loc='lower'):

    choelsky_rho_dict = {ifo: get_choelsky_decomp_of_toeplitz(rho) for ifo, rho in full_likelihood_manager.rho_dict.items()}
    whitened_data_dict = whiten_waveform_dict(full_likelihood_manager.data_dict, choelsky_rho_dict)
    whitened_reference_wf = whiten_waveform_dict(reference_waveform_dict, choelsky_rho_dict)

    time_dict = {ifo: times - full_likelihood_manager.reference_time for ifo, times in full_likelihood_manager.time_dict.items()}
    cp = sns.color_palette('muted')
    cp2 = sns.color_palette('pastel')

    dt_1M = 0.0127 / 10
    time_dict_M = {ifo: time / dt_1M for ifo, time in time_dict.items()}
    filenames = []
    for j, tc in enumerate(tc_floats):
        tc_M = tc / dt_1M

        # Make figure
        fig, axes = plt.subplots(1, 3, figsize=(15 / 1.3, 25 / 8 / 1.3))

        for ax in axes:
            ax.set_rasterization_zorder(2)

        # String for labeling the time step
        tc_str = f"{tc_M:.2f}" + 'M'  # tc_to_plot[j]
        lbl = tc_str.replace('m', '-') if tc_str[0] == 'm' else tc_str
        lbl = lbl.replace('M', '\,M_\mathrm{ref}')

        # Middle plots: bandpassed waveforms with cutoff
        x_lims = [-100, 70]
        # x_lims = [min(time_dict[ifo]), max(time_dict[ifo])]

        for k in [1, 2]:
            # Set up double axes
            top_ax = axes[k].twiny()
            top_ax.plot(time_dict[ifo], full_likelihood_manager.data_dict[ifo], color='k', alpha=0, zorder=3)
            top_ax.set_xlabel(r'$t~[s] $', fontsize=16, labelpad=10)
            top_ax.set_xlim(x_lims[0] * dt_1M, x_lims[1] * dt_1M)
            top_ax.grid(visible=False)

            axes[k].plot(time_dict_M[ifo], full_likelihood_manager.data_dict[ifo] / (1e-22), color='k', alpha=0, zorder=3)

            axes[k].plot(time_dict_M[ifo], reference_waveform_dict[ifo] / (1e-22), color='k')
            axes[k].axvline(tc_M, ls='--', color='k', zorder=4)

            axes[k].set_xlabel(r'$t~[M_\mathrm{ref}]$', fontsize=16)

            # Plot reconstructions before/after the cutoff
            n_reconstruction_to_plot = 300
            alpha_reconstruct = 0.2  # 0.02
            shading_kws = dict(hatch='/', alpha=0.3, color='gray')
            if k == 1:
                try:
                    wf_dict_list = wfs_at_tc_list[j]['pre']
                except KeyError:
                    wf_dict_list = []
                    
                for wf_dict in wf_dict_list:
                    axes[k].plot(time_dict_M[ifo], wf_dict[ifo] / (1e-22),
                                 color=cp[0], alpha=alpha_reconstruct, zorder=1)
                axes[k].axvspan(tc_M, x_lims[1], **shading_kws, zorder=0)
            else:
                try:
                    wf_dict_list = wfs_at_tc_list[j]['post']
                except KeyError:
                    wf_dict_list = []
                for wf_dict in wf_dict_list:
                    axes[k].plot(time_dict_M[ifo], wf_dict[ifo] / (1e-22),
                                 color=cp[1], alpha=alpha_reconstruct, zorder=1)
                axes[k].axvspan(x_lims[0], tc_M, **shading_kws, zorder=0)

            axes[k].grid(color='silver', ls=':', alpha=0.7)
            axes[k].set_xlim(*x_lims)
            axes[k].set_ylim(-7, 7)

            # Add label for cutoff time
            if tc_M >= 20:
                textloc = tc_M - 43
            elif tc_M == 17.5:
                textloc = tc_M - 53
            else:
                textloc = tc_M + 4
            axes[k].text(textloc, 5, f'${lbl}$', color='k', fontsize=15, zorder=5)

        axes[2].yaxis.set_label_position("right")
        axes[2].yaxis.tick_right()
        axes[2].set_ylabel(r'$h/10^{-22}$', fontsize=16)
        axes[1].set_yticklabels([])

        dx = -0.05
        x0, y0, x1, y1 = axes[1].get_position().bounds
        axes[1].set_position([x0 + dx + 0.0375, y0, x1, y1])
        x0, y0, x1, y1 = axes[2].get_position().bounds
        axes[2].set_position([x0 + dx, y0, x1, y1])

        # Righthand plots, plot posteriors
        try:
            axes[0].hist(dfs_at_tc[j]['pre'][plot_param], histtype='step', bins=30, lw=1.5,
                         color=cp[0], density=True)
        except KeyError:
            pass

        try:
            axes[0].hist(dfs_at_tc[j]['post'][plot_param], histtype='step', bins=30, lw=1.5,
                         color=cp2[1], density=True)
        except KeyError:
            pass
        axes[0].hist(full_df[plot_param], histtype='step', bins=30, lw=1.5,
                     color='k', density=True)

        prior_kws = dict(color='dimgray', ls=':')
        if prior_df is not None:
            axes[0].hist(prior_df[plot_param], histtype='step', bins=30, lw=1.5,
              density=True, **prior_kws)
        axes[0].set_xlabel(plot_param, fontsize=16)
        axes[0].set_ylabel(f"p({plot_param})", fontsize=16)

        # Inset axis in leftmost column with whitened strain
        l = 0.4
        w = 0.3
        dx = 0.04
        axin = axes[0].inset_axes(get_inset_axes_position(axes[0], dx, l, w, loc=loc))
        axin.plot(time_dict_M[ifo], whitened_data_dict[ifo], color='silver', lw=0.5)
        axin.plot(time_dict_M[ifo], whitened_reference_wf[ifo], color='k', lw=0.75)
        axin.axvline(tc_M, ls='--', color='k')
        axin.axvspan(tc_M, x_lims[-1], color=cp2[1], alpha=0.3, zorder=0)
        axin.axvspan(x_lims[0], tc_M, color=cp[0], alpha=0.3, zorder=0)
        axin.set_xlim(-60, 70)
        # axin.set_ylim(-3, 3)
        axin.set_xticklabels([])
        axin.set_yticklabels([])

        for ax in axes:
            x0, y0, x1, y1 = ax.get_position().bounds
            ax.set_position([x0, y0, x1, y1 ])#+ 0.05 / len(tc_floats)])

        # Legend
        handles = [
            Line2D([], [], color=cp[0], label='pre-cutoff analysis'),
            Line2D([], [], color=cp2[1], label='post-cutoff analysis'),
            Line2D([], [], color='k', label='full analysis'),
            Line2D([], [], **prior_kws, label='prior')
        ]
        leg = axes[2].legend(
            handles=handles,
            bbox_to_anchor=(-2.15, 1.4, 3.2, .102), loc='lower left', ncols=4,
            mode="expand", borderaxespad=0., frameon=False, fontsize=14, handlelength=3
        )
        for i, h in enumerate(handles):
            leg.get_lines()[i].set_linewidth(3)

        if save_dir is not None:
            try:
                os.mkdir(save_dir)
            except:
                pass
            filename = f'{save_dir}/frame_{tc_str}.png'
            plt.savefig(filename, bbox_inches='tight', dpi=200)
            filenames.append(filename)
        #
        # if j == 24:
        #     plt.show()
        # else:
        #     plt.close()
    
    if save_dir is not None:
        frames = []
        for filename in filenames:
            image = imageio.v2.imread(filename)
            frames.append(image)
        fps = 1.5
        imageio.mimsave(f'{save_dir}/all.gif', frames, fps=fps) 


def get_inset_axes_position(ax, dx, l, w, loc='upper'):
    
    if loc == 'upper':
        return dx - 0.01, 1 - dx - w, l, w  # Place the inset axes at the bottom left if the legend is at the top
    if loc == 'lower':
        return dx - 0.01, dx, l, w  # Place the inset axes at the top left if the legend is at the bottom
