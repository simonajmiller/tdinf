# Example: Running `TDinf` on GW190521


## 🔹 Step 1: Download the data
To run this example, the necessary strain data, PSDs, and reference posteriors must be downloaded. Do this with:
```
cd GW190521_input_data
chmod +x get_data.sh
./get_data.sh
```

## 🔹 Step 2: Run `TDinf`

### Option 1: Run `TDinf` using `condor` or `slurm` (Preferred)
We pass the [`GW190521.ini`](https://github.com/simonajmiller/time-domain-gw-inference/blob/main/examples/GW190521/GW190521.ini) config file to one of scripts in the [`pipe`](https://github.com/simonajmiller/time-domain-gw-inference/tree/main/pipe) folder to run `TDinf` on a computing cluster. The `.ini` file provides paths to the input data, plus sampler and waveform settings. We additionally tell the `pipe` script which cutoff times and/or cycles to run on. The `pipe` script then creates an output directory for the run results, which includes sub-folders for all cutoff times, plus copies of the input data, config file, and all necessary commands needed to reproduce results. 

| Cluster manager | Python script |  How to run in this example | 
| :---- | :---------------- | :------ | 
| condor | [`time_domain_inference_condor_pipe.py`](https://github.com/simonajmiller/time-domain-gw-inference/blob/main/pipe/time_domain_gw_inference_condor_pipe.py)|  [`./GW190521_pipe_condor.sh`](https://github.com/simonajmiller/time-domain-gw-inference/blob/main/examples/GW190521/GW190521_pipe_condor.sh) |
| slurm | [`time_domain_inference_slurm_pipe.py`](https://github.com/simonajmiller/time-domain-gw-inference/blob/main/pipe/time_domain_gw_inference_slurm_pipe.py)    |  [`./GW190521_pipe_slurm.sh`](https://github.com/simonajmiller/time-domain-gw-inference/blob/main/examples/GW190521/GW190521_pipe_slurm.sh) |

To check progress mid-run, you can run: 
```
tail -f output/full_0.0seconds/full_0.0seconds.log
```
(or replace with any of the run names, e.g, `post_-0.05seconds/post_-0.05seconds.log`)

> [!WARNING]
> Time-domain inference is computationally intensive. One `TDinf` run typically takes 8-12 hours to finish on 128 CPUs. 

### Option 2: Run `TDinf` in the commandline

If you do not have access to a computing cluster, `TDinf` can also be in the commandline, although it is ***not recommended*** due to computational expense and because it is not compatible with `group_postprocess.py` or `waveform_h5s.py` which aid in easily loading and plotting results. The script [`./commandline_example.sh`](https://github.com/simonajmiller/time-domain-gw-inference/blob/main/examples/GW190521/commandline_example.sh) shows an example for running in the commandline. 

## 🔹 Step 2.5: Generate waveform reconstructions from posteriors

If you use the provided `GW190521.ini` config file, waveform reconstructions from the posteriors for each run (full, pre- and post-cutoff) will be automatically generated after inference finishes. 
This can also be done manually with [`./generate_waveforms.sh`](https://github.com/simonajmiller/time-domain-gw-inference/blob/main/examples/GW190521/generate_waveforms.sh).

## 🔹 Step 3: View results

See [`GW190521_plot_results.ipynb`](https://github.com/simonajmiller/time-domain-gw-inference/blob/main/examples/GW190521/GW190521_plot_results.ipynb) for how to load and plot the data and results using the `group_postprocess` and `waveform_h5s` modules.

If you have any questions or concerns about running this example, or about `TDinf` in general, please email `simona.miller@ligo.org`. 📩
