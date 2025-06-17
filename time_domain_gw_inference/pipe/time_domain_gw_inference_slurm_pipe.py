#!/usr/bin/env python3
# coding: utf-8
"""
Generate and optionally submit a Slurm + disBatch workflow for the time_domain_gw_inference pipeline.
"""

import sys
import os
import argparse
import logging
import configparser
import subprocess
import shutil
import ast

def get_parser():
    p = argparse.ArgumentParser(description="time_domain_gw_inference Slurm + disBatch pipeline")
    p.add_argument("--config_file", required=True, help="Path to the configuration file")
    p.add_argument("--output_directory", required=True, help="Output directory for workflow")
    p.add_argument("--cycle_list", nargs='+', type=float, help="Cycles before merger to cut data at")
    p.add_argument("--times_list", nargs='+', type=float, help="Times before merger to cut data at")
    p.add_argument("--modes", nargs='+', type=str, default=("full","pre","post"), help="Run full, pre, and/or post?")
    p.add_argument("--submit", action="store_true", help="Submit the workflow to Slurm")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing directory")
    p.add_argument("--ntasks", type=int, default=100, help="Maximum number of tasks to request through SLURM")
    p.add_argument("--partition", type=str, default='cca', help="Partition to run")
    p.add_argument("--constraints", help="SLURM constraints")
    p.add_argument("--time", help="SLURM time directive")
    return p

def copy_file_to_directory_and_return_new_name(file, target_directory, relative_path=None):
    # Copy the file to the target directory
    shutil.copy(file, target_directory)
    # Get just the filename without the folder its in
    just_filename = os.path.basename(file)
    # Path to new file
    new_file_path = os.path.join(target_directory, just_filename)
    if relative_path is not None:
        new_file_path = os.path.relpath(new_file_path, relative_path)
    return new_file_path

def main(args=None):

    # Parse arguments
    parser = get_parser()
    args = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO if args.submit else logging.WARNING)

    # Load config
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.optionxform = str
    config.read(args.config_file)
    executables = dict(config.items("executables"))
    run_settings = dict(config.items("time_domain_gw_inference"))
    if 'waveform_h5s' in config.sections(): 
        wf_settings = dict(config.items("waveform_h5s"))
    else: 
        wf_settings = dict()

    # Prepare output directory
    outdir = os.path.abspath(args.output_directory)
    if os.path.exists(outdir):
        if args.overwrite:
            logging.warning(f"Output directory {outdir} already exists. Overwriting.")
            shutil.rmtree(outdir)
            os.makedirs(outdir)
        else: 
            raise ValueError(f"Output directory {outdir} already exists. Quitting.")
    else:
        os.makedirs(outdir)

    # Copy config for reproducibility
    shutil.copy(args.config_file, os.path.join(outdir, "config.ini"))

    # Record command
    with open(os.path.join(outdir, "command.sh"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    # Make `data_directory` for run input data
    data_directory = os.path.join(outdir, 'data_directory')
    os.makedirs(data_directory, exist_ok=True)

    # Copy over the reference parameters and/or posterior into `data_directory` 
    for f in ['injected-parameters', 'reference-parameters', 'pe-posterior-h5-file']: 
        fpath = run_settings.pop(f, None)
        if fpath is not None: 
            run_settings[f] = copy_file_to_directory_and_return_new_name(
                fpath, data_directory, args.output_directory
            )

    # Extract strain data path dict and PSD path dict
    data_dict_str = run_settings.pop('data-path-dict')
    psd_dict_str = run_settings.pop('psd-path-dict')

    # Make commandline strings          
    run_options = ''.join([f'--{k} {v} ' for k,v in run_settings.items()])
    wf_options = ''.join([f'--{k} {v} ' for k,v in wf_settings.items()])

    # Parse strain datas paths info, copy to data_directory, and add to the run_options
    data_dict = {ifo:copy_file_to_directory_and_return_new_name(
            fpath, data_directory, args.output_directory
        ) for ifo, fpath in ast.literal_eval(data_dict_str).items()}
    run_options += ''.join([f"--data {k}:{v} " for k, v in data_dict.items()])

    # Parse PSD paths info, copy to data_directory, and add to the run_options
    psd_dict = {ifo:copy_file_to_directory_and_return_new_name(
            fpath, data_directory, args.output_directory
        ) for ifo, fpath in ast.literal_eval(psd_dict_str).items()}
    run_options += ''.join([f"--psd {k}:{v} " for k, v in psd_dict.items()])

    # Prepare tasks files for each pipeline stage
    tasks_run = os.path.join(outdir, "tasks_run.txt")
    tasks_wave = os.path.join(outdir, "tasks_waveforms.txt")
    with open(tasks_run, "w") as tf_run, open(tasks_wave, "w") as tf_wave:  

        # Cutoff times list and /or cutoff cycles list
        cycle_list = args.cycle_list or []
        times_list = args.times_list or []

        def get_modes(val):
            if 'full' in args.modes and val==0: 
                return args.modes
            else: 
                return [k for k in args.modes if k!="full"]
                
        def run_label(mode, cut, unit):
            return f"{mode}_{cut}{unit}"
            
        # Build tasks for cycles
        for cut in cycle_list:
            modes = get_modes(cut)
            for mode in modes:
                run_lbl = run_label(mode, cut, "cycles")
                # make directory for run output
                os.makedirs(os.path.join(outdir, run_lbl), exist_ok=True)
                # run_sampler command
                run_cmd = [executables["run_sampler"], "--output-h5", f"{run_lbl}/{run_lbl}.h5", "--mode", mode, 
                           "--Tcut-cycles", str(cut), run_options, f"&>> {run_lbl}/{run_lbl}.log"]
                tf_run.write(" ".join(run_cmd) + "\n")
                # make_waveforms
                if "waveform_h5s" in executables:
                    wf_cmd = [executables["waveform_h5s"], "--directory", run_lbl, "--run_key", run_lbl, wf_options]
                    tf_wave.write(" ".join(wf_cmd) + "\n")

        # Build tasks for times
        for cut in times_list:
            modes = get_modes(cut)
            for mode in modes:
                run_lbl = run_label(mode, cut, "seconds")
                # make directory for run output
                os.makedirs(os.path.join(outdir, run_lbl), exist_ok=True)
                run_cmd = [executables["run_sampler"], "--output-h5", f"{run_lbl}/{run_lbl}.h5", "--mode", mode, 
                           "--Tcut-seconds", str(cut), run_options, f"&>> {run_lbl}/{run_lbl}.log"]
                tf_run.write(" ".join(run_cmd) + "\n")
                # make_waveforms
                if "waveform_h5s" in executables:
                    wf_cmd = [executables["waveform_h5s"], "--directory", run_lbl, "--run_key", run_lbl, wf_options]
                    tf_wave.write(" ".join(wf_cmd) + "\n")

    # Create submission script with dependencies between stages
    NCPU = int(run_settings['ncpu'])
    submit_sh = os.path.join(outdir, "submit.sh")
    with open(submit_sh, "w") as sf:
        sf.write("#!/usr/bin/env bash\n")
        sf.write("function get_id() {\n")
        sf.write("    if [[ \"$1\" =~ Submitted\\ batch\\ job\\ ([0-9]+) ]]; then\n")
        sf.write("        echo \"${BASH_REMATCH[1]}\"\n")
        sf.write("    else\n")
        sf.write("        echo \"sbatch failed\"\n")
        sf.write("        exit 1\n")
        sf.write("    fi\n")
        sf.write("}\n\n")
        sf.write(f"cd {outdir}\n")
        sb = "sbatch"
        if args.constraints:
            sb += f" -C {args.constraints}"
        if args.time:
            sb += f" -t {args.time}"
        # Stage 1: run_sampler
        sf.write(f'runid=$(get_id "$({sb} -p {args.partition} -n {args.ntasks} -c {NCPU} disBatch {tasks_run})")\n')
        # Stage 2: make_waveforms
        sf.write(f'waveid=$(get_id "$({sb} --dependency=afterok:$runid -p {args.partition} -n 8 -c {NCPU} disBatch {tasks_wave})")\n')
        sf.write("cd -\n")

    os.chmod(submit_sh, 0o755)
    if args.submit:
        subprocess.run([submit_sh])
    else:
        print(f"\n**********************************************************************")
        print(f"  Submit the workflow with: {submit_sh}")
        print(f"**********************************************************************\n")

if __name__ == "__main__":
    main()