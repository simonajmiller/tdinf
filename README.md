# time-domain-gw-inference

`review` branch is for review of the time domain inference code for S240114ax analysis. 

to use the code: clone the repository and from the root run 
```
pip install -e .
```

then run the following executable. this is an example of running on S2501124ax on `simona.miller`'s CIT account, for full and pre/post $t=0$.
```
./example_runpipe.sh
```
this will set up the run and print out the commandline to submit the relevant dag file. in the example this outputs: 
```
******************************************************
To submit the DAG, run the following command:
	condor_submit_dag -import_env -usedagdir /home/simona.miller/arealawrescue/time_domain_analysis/output/test/test.dag
******************************************************
```
 
note: exclues everything related to the eccentric analyses. 
 
**do not merge this branch into `main`**
