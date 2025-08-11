# Example: Running `TDinf` on GW190521


### Step 1: Download the data
To run this example, the necessary strain data, PSDs, and reference posteriors must be downloaded. Do this with:
```
cd GW190521_input_data
chmod +x get_data.sh
./get_data.sh
```

### Step 2: Run `TDinf` using `condor` or `slurm`

We pass the [`GW190521.ini`](https://github.com/simonajmiller/time-domain-gw-inference/blob/main/examples/GW190521/GW190521.ini) config file to one of scripts in the [`pipe`](https://github.com/simonajmiller/time-domain-gw-inference/tree/main/pipe) folder to run `TDinf` on a computing cluster. The `.ini` file provides paths to the input data, plus sampler and waveform settings. We additionally tell the `pipe` script which cutoff times and/or cycles to run on. The `pipe` script then creates an output directory for the run results, which includes sub-folders for all cutoff times, plus copies of the input data, config file, and all necessary commands needed to reproduce results. 

| Cluster manager | Python script |  How to call in this example | 
| :---- | :---------------- | :------ | 
| condor | [`time_domain_inference_condor_pipe.py`](https://github.com/simonajmiller/time-domain-gw-inference/blob/main/pipe/time_domain_gw_inference_condor_pipe.py)|  [`./GW190521_pipe_condor.sh`](https://github.com/simonajmiller/time-domain-gw-inference/blob/main/examples/GW190521/GW190521_pipe_condor.sh) |
| slurm | [`time_domain_inference_slurm_pipe.py`](https://github.com/simonajmiller/time-domain-gw-inference/blob/main/pipe/time_domain_gw_inference_slurm_pipe.py)    |  [`./GW190521_pipe_slurm.sh`](https://github.com/simonajmiller/time-domain-gw-inference/blob/main/examples/GW190521/GW190521_pipe_slurm.sh) |

To check progress mid-run, you can run: 
```
tail -f output/full_0.0seconds/full_0.0seconds.log
```
(or replace with any of the run names, e.g, `post_-0.05seconds/post_-0.05seconds.log`)

> [!WARNING]
> Time-domain inference is computationally intensive. One `TDinf` run typically takes O(8-12 hours) to finish on 128 CPUs. 

### Step 2.5: Generate waveform reconstructions from posteriors

If you use the provided `GW190521.ini` config file, waveform reconstructions from the posteriors for each run (full, pre- and post-cutoff) will be automatically generated after inference finishes. 
This can also be done manually with [`./generate_waveforms.sh`](https://github.com/simonajmiller/time-domain-gw-inference/blob/main/examples/GW190521/generate_waveforms.sh).

### Step 3: See [`GW190521_plot_results.ipynb`](https://github.com/simonajmiller/time-domain-gw-inference/blob/main/examples/GW190521/GW190521_plot_results.ipynb) for how to plot load and plot the data and results.
