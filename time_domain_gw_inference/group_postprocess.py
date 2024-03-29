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
            except TypeError as e:
                print(e)
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
                                          vary_eccentricity=args.vary_eccentricity, 
                                          use_higher_order_modes=args.use_higher_order_modes, **kwargs)
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
        use_higher_order_modes=args.use_higher_order_modes,
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

def get_tc_from_name(name):
    pattern = re.compile(r'(post|pre|full)_(-?\d+\.\d+)seconds')
    match = pattern.match(name)
    if match:
        prefix, nseconds = match.groups()
    else:
        raise ValueError(f"name {name} did not match regex, could not find name")
    return nseconds


def generate_filename_dict(directory):
    filename_dict = {}
    pattern = re.compile(r'(post|pre|full)_(-?\d+\.\d+)seconds')

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
import copy


def whiten_waveform_dict(wf_dict, likelihood_manager):
    whitened_wf_dict = {}
    for ifo in wf_dict.keys():
        whitened_wf_dict[ifo] = utils.whitenData(wf_dict[ifo], likelihood_manager.time_dict[ifo],
                                                 likelihood_manager.psd_dict[ifo][:, 1], likelihood_manager.psd_dict[ifo][:, 0])
    return whitened_wf_dict

def get_choelsky_decomp_of_toeplitz(toeplitz_matrix):
    # compute covariance matrix  C and its Cholesky decomposition L (~sqrt of C)
    C = scipy.linalg.toeplitz(toeplitz_matrix)
    L = np.linalg.cholesky(C)
    return L


def make_inset_plot(ax, times, data_wf, reference_wf, t_cut, 
                    l=0.4, w=0.3, dx=0.04, xlim=None, inset_ylim=None, **kwargs):
    cp = sns.color_palette('muted')
    cp2 = sns.color_palette('pastel')
    
    # Inset axis in leftmost column with whitened strain
    axin = ax.inset_axes(get_inset_axes_position(ax, dx, l, w, loc=kwargs.get('loc', 'upper')))
    axin.plot(times, data_wf, color='silver', lw=0.5)
    axin.plot(times, reference_wf, color='k', lw=0.75)
    
    if xlim is None:
        xlim = axin.get_xlim()

    axin.axvline(t_cut, ls='--', color='k', zorder=0)
    axin.axvspan(t_cut, xlim[-1], color=cp2[1], alpha=0.3, zorder=1)
    axin.axvspan(xlim[0], t_cut, color=cp[0], alpha=0.3, zorder=1)
    axin.set_xlim(xlim)
    if inset_ylim is None:
        inset_ylim = axin.get_ylim()
    axin.set_ylim(inset_ylim)
    axin.set_xticklabels([])
    axin.set_yticklabels([])


def time_to_mass(times):
    dt_1M = 0.0127 / 10
    return times / dt_1M


def make_gif(tc_floats, wfs_at_tc_list, reference_waveform_dict, dfs_at_tc, full_df,
             full_likelihood_manager,
             plot_param, prior_df = None, 
             ifo='L1', save_dir = None, loc='lower', 
             ylim=None, inset_ylim=None, 
             xlim=None, 
             param_xlim=None):

    whitened_data_dict = whiten_waveform_dict(full_likelihood_manager.data_dict, full_likelihood_manager)
    whitened_reference_wf = whiten_waveform_dict(reference_waveform_dict, full_likelihood_manager)

    time_dict = {_ifo: times - full_likelihood_manager.reference_time  for _ifo, times in full_likelihood_manager.time_dict.items()}
    cp = sns.color_palette('muted')
    cp2 = sns.color_palette('pastel')

    time_dict_M = {_ifo: time_to_mass(time) for _ifo, time in time_dict.items()}
    filenames = []
    for j, tc in enumerate(tc_floats):
        tc_M = time_to_mass(tc)

        # Make figure
        fig, axes = plt.subplots(1, 3, figsize=(15 / 1.3, 25 / 8 / 1.3))

        for ax in axes:
            ax.set_rasterization_zorder(2)

        # String for labeling the time step
        tc_str = f"{tc_M:.2f}" + 'M'  # tc_to_plot[j]
        lbl = tc_str.replace('m', '-') if tc_str[0] == 'm' else tc_str
        lbl = lbl.replace('M', '\,M_\mathrm{ref}')

        # Middle plots: bandpassed waveforms with cutoff
        if xlim is None:
            xlim = (-100, 70)
        # x_lims = [min(time_dict[ifo]), max(time_dict[ifo])]

        

        for k in [1, 2]:
            # Set up double axes
            top_ax = axes[k].twiny()
            top_ax.plot(time_dict[ifo], full_likelihood_manager.data_dict[ifo], color='k', alpha=0, zorder=3)
            top_ax.set_xlabel(r'$t~[s] $', fontsize=16, labelpad=10)
            #top_ax.set_xlim(xlim[0] * dt_1M, xlim[1] * dt_1M)
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
                axes[k].axvspan(tc_M, xlim[1], **shading_kws, zorder=0)
            else:
                try:
                    wf_dict_list = wfs_at_tc_list[j]['post']
                except KeyError:
                    wf_dict_list = []
                for wf_dict in wf_dict_list:
                    axes[k].plot(time_dict_M[ifo], wf_dict[ifo] / (1e-22),
                                 color=cp[1], alpha=alpha_reconstruct, zorder=1)
                axes[k].axvspan(xlim[0], tc_M, **shading_kws, zorder=0)

            axes[k].grid(color='silver', ls=':', alpha=0.7)
            axes[k].set_xlim(xlim)
            if ylim is None:
                ylim = axes[k].get_ylim()
            axes[k].set_ylim(ylim[0], ylim[1])

            # Add label for cutoff time
            if tc_M >= 20:
                textloc = tc_M - 43
            elif tc_M == 17.5:
                textloc = tc_M - 53
            else:
                textloc = tc_M + 4
                
            yrange = ylim[1] - ylim[0]
            
            axes[k].text(textloc, ylim[1] - 0.20 * yrange, f'${lbl}$', color='k', fontsize=15, zorder=5)

        axes[2].yaxis.set_label_position("right")
        axes[2].yaxis.tick_right()
        axes[2].set_ylabel(r'$h/10^{-22}$', fontsize=16)
        axes[1].set_yticklabels([])

        dx = -0.05
        x0, y0, x1, y1 = axes[1].get_position().bounds
        axes[1].set_position([x0 + dx + 0.0375, y0, x1, y1])
        x0, y0, x1, y1 = axes[2].get_position().bounds
        axes[2].set_position([x0 + dx, y0, x1, y1])
        
        # -------------------------------------------------------------------------
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
        if param_xlim is None: 
            param_xlim = axes[0].get_xlim()
        axes[0].set_xlim(param_xlim)
        # -------------------------------------------------------------------------


        # Inset axis in leftmost column with whitened strain
        l = 0.4
        w = 0.3
        dx = 0.04
        axin = axes[0].inset_axes(get_inset_axes_position(axes[0], dx, l, w, loc=loc))
        axin.plot(time_dict_M[ifo], whitened_data_dict[ifo], color='silver', lw=0.5)
        axin.plot(time_dict_M[ifo], whitened_reference_wf[ifo], color='k', lw=0.75)
        # axin.plot(time_dict_M[ifo], full_likelihood_manager.data_dict[ifo], color='silver', lw=0.5)
        # axin.plot(time_dict_M[ifo], reference_waveform_dict[ifo], color='k', lw=0.75)

        axin.axvline(tc_M, ls='--', color='k', zorder=0)
        axin.axvspan(tc_M, xlim[-1], color=cp2[1], alpha=0.3, zorder=1)
        axin.axvspan(xlim[0], tc_M, color=cp[0], alpha=0.3, zorder=1)
        axin.set_xlim(xlim)
        if inset_ylim is None:
            inset_ylim = axin.get_ylim()
        axin.set_ylim(inset_ylim)
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
            
    gif_name = f'{save_dir}/all.gif'
    if save_dir is not None:
        save_gif(filenames, gif_name, fps=1.5)

def save_gif(filenames, gif_name, fps=1.5):
    frames = []
    for filename in filenames:
        image = imageio.v2.imread(filename)
        frames.append(image)
    fps = 1.5
    imageio.mimsave(gif_name, frames, fps=fps) 


def get_inset_axes_position(ax, dx, l, w, loc='upper'):
    
    if loc == 'upper':
        return dx - 0.01, 1 - dx - w, l, w  # Place the inset axes at the bottom left if the legend is at the top
    if loc == 'lower':
        return dx - 0.01, dx, l, w  # Place the inset axes at the top left if the legend is at the bottom


def make_color_dict(keys):
    # Initialize the color dictionary
    color_dict_inj = {}
    if 'full' in keys:
        color_dict_inj['full'] = 'black'
    
    # Define the color map for shades of blue and orange
    matplotlib.colors.LinearSegmentedColormap.from_list("",['#ed9911', #'#995c00', 
                                                            '#ffad33'])
    blue_map = matplotlib.colors.LinearSegmentedColormap.from_list("",[ '#4db8ff', '#005c99'])  #cm.Blues
    orange_map = matplotlib.colors.LinearSegmentedColormap.from_list("",['#995c00',  '#ffc266']) #cm.Oranges
    #blue_map = cm.Blues
    #orange_map = cm.Oranges
    
    # Extract unique numbers from 'pre' and 'post' keys
    pre_keys = sorted([(key, float(key.split('_')[1])) for key in keys if key.startswith('pre')], key=lambda x: x[1])
    post_keys = sorted([(key, float(key.split('_')[1])) for key in keys if key.startswith('post')], key=lambda x: x[1])

    pre_range = max([pre_val for _, pre_val in pre_keys]) - min([pre_val for _, pre_val in pre_keys])
    post_range = max([val for _, val in post_keys]) - min([val for _, val in post_keys])
    
    norm_pre = matplotlib.colors.Normalize(vmin=min([val for _, val in pre_keys]), 
                                                    vmax=max([val for _, val in pre_keys]))
    norm_post = matplotlib.colors.Normalize(vmin=min([val for _, val in post_keys]), 
                                                    vmax=max([val for _, val in post_keys]))

    # Assign shades of blue for 'pre' keys
    for key, num in pre_keys:
        color_dict_inj[key] = matplotlib.colors.to_hex(blue_map(norm_pre(num)))
    for key, num in post_keys:
        color_dict_inj[key] = matplotlib.colors.to_hex(orange_map(norm_pre(num)))
    return color_dict_inj


def get_non_constant_keys(df):
    plot_keys = []
    for key in df.keys():
        if max(df[key]) == min(df[key]):
            continue
        plot_keys.append(key)
    return plot_keys


def plot_corner(run_set, reference_df_key = 'full',
                plot_datapoints=False, plot_density=False, plot_contours=True, 
                run_keys=None, truth_dict=None, plot_keys=None, fig=None, 
                save_dir=None, **kwargs):
    
    hist2d_kwargs=dict(plot_datapoints=plot_datapoints, 
                       plot_density=plot_density, 
                       plot_contours=plot_contours)
    hist2d_kwargs.update(kwargs)
    td_samples = run_set['dfs']
    
    df = td_samples[reference_df_key]

    if plot_keys is None:
        plot_keys = get_non_constant_keys(df)
        
    if run_keys is None:
        run_keys = td_samples.keys()

    if truth_dict is not None:
        truths = [truth_dict[plot_key] for plot_key in plot_keys]
    else:
        truths = None
        
    fig = corner.corner(df, var_names=plot_keys, hist_kwargs=dict(density=True), 
                  fig=fig,
                  truths = truths, 
                  **hist2d_kwargs)
    return fig

def get_sorted_pre_and_post_keys(keys):
    t_cuts = sorted(list(set(float(key.split('_')[1]) if '_' in key else 0 for key in keys )))
    # Extract unique numbers from 'pre' and 'post' keys
    pre_keys = sorted([(key, float(key.split('_')[1])) for key in keys if key.startswith('pre')], key=lambda x: x[1])
    post_keys = sorted([(key, float(key.split('_')[1])) for key in keys if key.startswith('post')], key=lambda x: x[1])
            
    tcut_dict = {}

    # Iterate over unique t_cuts
    for t_cut in t_cuts:
        # Find the corresponding 'pre' key
        pre_key = next((key for key, t in pre_keys if t == t_cut), None)
        # Find the corresponding 'post' key
        post_key = next((key for key, t in post_keys if t == t_cut), None)
        
        # Create a dictionary entry for the current t_cut
        new_keys = []
        if pre_key is not None:
            new_keys.append(pre_key)
        if post_key is not None:
            new_keys.append(post_key)
        tcut_dict[t_cut] = new_keys
        
    return tcut_dict


def make_corner_gif(individual_run,
                    full_likelihood_manager, 
                    reference_waveform_dict,
                    ifo='L1',
                    run_keys=None, reference_parameters=None,
                    save_dir=None,
                    levels=1, range_dict=None, prior_df=None, **kwargs):

    if run_keys is None:
        run_keys = individual_run['dfs'].keys()
    
    tcut_dict = get_sorted_pre_and_post_keys(run_keys)

    whitened_data_dict = whiten_waveform_dict(full_likelihood_manager.data_dict, full_likelihood_manager)
    whitened_reference_wf = whiten_waveform_dict(reference_waveform_dict, full_likelihood_manager)

    time_dict = {_ifo: times - full_likelihood_manager.reference_time  for _ifo, times in full_likelihood_manager.time_dict.items()}
    time_dict_M = {_ifo: time_to_mass(time) for _ifo, time in time_dict.items()}

    plot_keys = kwargs.get('plot_keys', None)
    if plot_keys is None:
        plot_keys = get_non_constant_keys(df)

    plot_kwargs = {}

    if range_dict is not None:
        if prior_df is not None:
            reference_df = prior_df
        else:
            reference_df = individual_run['dfs']['full']
        
        plot_kwargs['range'] = [range_dict.get(key, (min(reference_df[key]), max(reference_df[key]))) for key in plot_keys]
        
    color_dict = make_color_dict(individual_run['dfs'].keys())
    hist_kwargs = dict( histtype='step', density=True, bins='auto')

    filenames = []

    if save_dir is not None:
        try:
            os.mkdir(save_dir)
        except:
            pass

    ylims = [None for key in plot_keys]
    for t_cut, keys in tcut_dict.items():
        
        
        plot_kwargs['fig'] = None
        plot_kwargs['contour_kwargs'] = {'levels':levels}
        fig = plot_corner(individual_run, reference_df_key = 'full',
                          truth_dict=reference_parameters,
                          color='black', 
                          truth_color='black',
                          **plot_kwargs, **kwargs)
            
        for key in keys:
            plot_kwargs['fig'] = fig
            plot_kwargs['contour_kwargs'] = {'levels':1}
            fig = plot_corner(individual_run, reference_df_key = key,
                          color=color_dict[key],
                          **plot_kwargs, **kwargs)

    
        axes = np.array(fig.axes).reshape((len(plot_keys), len(plot_keys)))
        # corner does weird shit with histograms, replot here 
        print(ylims)
        for i in range(len(plot_keys)):
            parameter = plot_keys[i]
            ax = axes[i, i]
            artists = ax.patches

            for artist in artists:
                # Check if the artist is a polygon (histogram)
                if isinstance(artist, plt.Polygon): 
                    artist.remove()

            if ylims[i] is None:
                ax.autoscale(axis="y")
                
            ax.hist(individual_run['dfs']['full'][parameter],
                   color='black', **hist_kwargs)
            for key in keys:
                ax.hist(individual_run['dfs'][key][parameter],
                       color=color_dict[key], **hist_kwargs)
            if prior_df is not None:
                ax.hist(prior_df[parameter], linestyle='dotted',
                        color='silver', **hist_kwargs)
            
            if ylims[i] is None:
                ax.autoscale(False)
                ylim = ax.get_ylim()
                ylims[i] = (ylim[0], ylim[1] * 1.2)
            ax.set_ylim(ylims[i])

        ax = axes[0, -1]
        
        make_inset_plot(ax, time_dict_M[ifo], whitened_data_dict[ifo], whitened_reference_wf[ifo],
                        time_to_mass(t_cut), l=0.5, **kwargs)
        
        
        
        
        tc_str = f"{time_to_mass(t_cut):.2f}" + 'M'  # tc_to_plot[j]
        lbl = tc_str.replace('m', '-') if tc_str[0] == 'm' else tc_str
        lbl = lbl.replace('M', '\,M_\mathrm{ref}')
        
        ylim = ax.get_ylim()
        yrange = ylim[1] - ylim[0]
        xlim = ax.get_xlim()
        xrange = xlim[1] - xlim[0]
        textloc = xlim[0] + 0.2 * xrange
        ax.text(textloc, ylim[1] - 0.50 * yrange, f'${lbl}$', color='k', fontsize=15, zorder=5)
        
        for i in range(len(plot_keys)):
            ax = axes[i, i]
            ax.set_ylim(ylims[i])

        

        if save_dir is not None:
            filename = f'{save_dir}/corner_frame_{tc_str}.png'
            plt.savefig(filename, bbox_inches='tight', dpi=200)
            filenames.append(filename)
    if save_dir is not None:
        save_gif(filenames, f'{save_dir}/all_corner.gif')
    
    

    

