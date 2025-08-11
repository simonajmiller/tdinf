# Example: Running `TDinf` on GW190521


### Step 1: Download the data
```
cd GW190521_input_data
chmod +x get_data.sh
./get_data.sh
```

### Step 2: Run `TDinf` using `condor` or `slurm`

The `.ini` file contains all the info needed for a given run, *except* for the specific cutoff times desired and the path to the output. 

If using `condor` for cluster management, run:
```
chmod +x GW190521_pipe_condor.sh
./GW190521_pipe_condor.sh
```

If using with `slurm` for cluster management, run:
```
chmod +x GW190521_pipe_slurm.sh
./GW190521_pipe_slurm.sh
```

To check progress mid-run, you can run: 
```
tail -f output/full_0.0seconds/full_0.0seconds.log
```
(or replace with any of the run names, e.g, `post_-0.05seconds/post_-0.05seconds.log`)

### Step 2.5: Generate waveform reconstructions from posteriors

If you use the provided `GW190521.ini` config file, waveform reconstructions from the posteriors for each run (full, pre- and post-cutoff) will be automatically generated after inference finishes. 
This can also be done manually via:
```
chmod +x generate_waveforms.sh
./generate_waveforms.sh
```

### Step 3: See [`GW190521_plot_results.ipynb`](https://github.com/simonajmiller/time-domain-gw-inference/blob/main/examples/GW190521/GW190521_plot_results.ipynb) for how to plot load and plot the data and results.
