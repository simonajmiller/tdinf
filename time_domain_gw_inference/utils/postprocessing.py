import numpy as np
import lal
import lalsimulation as lalsim
import sys
import pandas as pd
import emcee

try: 
    from spins_and_masses import m1m2_from_mtotq
    import likelihood as ll
except: 
    from .spins_and_masses import m1m2_from_mtotq
    from . import likelihood as ll 
    
    
def get_dict_from_samples(samples, **kwargs):
    
    """
    Transform samples from their logistic space values to their physical values, 
    convert into the quantities we want (spin magnitudes and tilts, etc.), and
    wrap it all up in a dictionary 
    """
        
    # get physical parameters
    samps_phys = np.array([ll.samp_to_phys(samp, **kwargs) for samp in samples], ndmin=2)
                
    # change to LALInference spin convention
    ys = []
    for samp in samps_phys:
        mtot, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, dist_mpc, phi_ref, iota, ra, dec, psi, tgps_geocent = samp
        m1, m2 = m1m2_from_mtotq(mtot, q)    
        ys.append(lalsim.SimInspiralTransformPrecessingWvf2PE(
            iota, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, m1, m2,
            kwargs['f_ref'], phi_ref
        ))
                
    # Quantities we care about
    theta_jn, phi_jl, tilt1, tilt2, phi12, chi1, chi2 = np.array(ys, ndmin=2).T
    mtot, q, _, _, _, _, _, _, dist_mpc, phi_ref, iota, ra, dec, psi, tgps_geocent = np.array(samps_phys.T)
    
    # Format in dictionary
    keys = ['mtotal', 'q', 'chi1', 'chi2', 'phi_jl', 'tilt1', 'tilt2', 'phi12', 'dist', 
            'phase', 'theta_jn', 'iota', 'ra', 'dec', 'psi', 'tgps_geocent']
    
    vals = [mtot, q, chi1, chi2, phi_jl, tilt1, tilt2, phi12, dist_mpc,
            phi_ref, theta_jn, iota, ra, dec, psi, tgps_geocent]
    
    samples_dict = dict(zip(keys, vals))
    
    return samples_dict


def postprocess_samples(sampler, getRidOfFixed=False, **kwargs): 
    
    """
    Post-process emcee sample chains
    
    input: sampler = emcee sampler object
    """
    
    # Get autocorrelation time
    tau = sampler.get_autocorr_time(quiet=True)
    
    # Calculate amonut to burn in and thin by based off of autocorr time
    burnin = int(5 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    
    # Convert samples into their physical quantities
    samples_dict = get_dict_from_samples(samples, **kwargs)

    # Add posterior values the sample_dict
    samples_lnp = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
    samples_dict['ln_posterior'] = samples_lnp
    
    # Add prior values to the sample_dict
    samples_lnprior = np.asarray([ll.get_lnprior(x, **kwargs) for x in samples])
    samples_dict['ln_prior'] = samples_lnprior

    # Turn into data frame
    df = pd.DataFrame(samples_dict)
    
    # Get rid of the fixed parameters if we want
    if getRidOfFixed:
        df = df[[k for k, v in df.items() if v.min() != v.max()]]
    
    return df 
