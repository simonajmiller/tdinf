# time-domain-gw-inference
working repository for time domain parameter estimation for gravitational-wave signals


# Inputs 
Optional 
   - File containing injected parameters (TODO add waveform approx to json file ) (json)
Required 
   - hdf5 file with gwf frame data # TODO should not be required for simulated data 
   - psd(s) path formatted as column frequency and value (usually .dat) 

# Outputs 
- h5 file from emcee sampling 
- dat file with output samples

