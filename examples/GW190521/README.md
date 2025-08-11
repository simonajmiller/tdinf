# Example: Running `TDinf` on GW190521

***!! WIP !!***

Download the data: 
```
cd GW190521_input_data
chmod +x get_data.sh
./get_data.sh
```

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


### See `GW190521_plot_results.ipynb` for how to plot load and plot the data and results.