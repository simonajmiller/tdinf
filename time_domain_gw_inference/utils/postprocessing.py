import numpy as np
import lalsimulation as lalsim

import pandas

try: 
    from spins_and_masses import m1m2_from_mtotq
    import likelihood as ll
except: 
    from .spins_and_masses import m1m2_from_mtotq
    from . import likelihood as ll 
    
    
def get_dict_from_samples(samples, parameter_manager, **kwargs):
    
    """
    Transform samples from their logistic space values to their physical values, 
    convert into the quantities we want (spin magnitudes and tilts, etc.), and
    wrap it all up in a dictionary 
    """
    # get physical parameters (list of dictionary)
    samps_phys = [parameter_manager.samp_to_phys(samp) for samp in samples]

    for sample in samps_phys:
        m1, m2 = m1m2_from_mtotq(sample['total_mass'], sample['mass_ratio'])
        sample['theta_jn'], sample['phi_jl'], \
        sample['tilt1'], sample['tilt2'], sample['phi12'], \
        sample['spin1_magnitude'], sample['spin2_magnitude'] = lalsim.SimInspiralTransformPrecessingWvf2PE(
                sample['inclination'],
                sample['spin1_x'], sample['spin1_y'],  sample['spin1_z'],
                sample['spin2_x'], sample['spin2_y'], sample['spin2_z'],
                m1, m2,
                kwargs['f_ref'], sample['phase']
            )
    return pandas.DataFrame(samps_phys)


def postprocessing_get_complete_samples_dict(samples, samples_lnp, likelihood_manager, getRidOfFixed=False, **kwargs): 

    # convert samples into their physical quantities
    samples_dict = get_dict_from_samples(samples, likelihood_manager.log_prior, **kwargs)

    # Add posterior values the sample_dict
    samples_dict['ln_posterior'] = samples_lnp
    
    # Add prior values to the sample_dict
    samples_lnprior = np.asarray([likelihood_manager.log_prior.get_lnprior(x) for x in samples])
    samples_dict['ln_prior'] = samples_lnprior
    
    # Add likelihood values 
    samples_dict['ln_likelihood'] = samples_lnp - samples_lnprior
    
    # Finally, generate SNRs and add them to the samples 
    print('Calculating SNRs from posterior ...')
    SNRs_dict = likelihood_manager.get_SNRs(samples)
    for k in SNRs_dict: 
        samples_dict[k] = SNRs_dict[k]
        
    print(samples_dict.keys())

    # Get rid of the fixed parameters if we want
    if getRidOfFixed:
        samples_dict = samples_dict[[k for k, v in samples_dict.items() if v.min() != v.max()]]

    return samples_dict

def postprocess_samples(sampler, likelihood_manager, getRidOfFixed=False, **kwargs):
    """
    Post-process emcee sample chains
    
    input: sampler = emcee sampler object
    """
    
    print('\nPOSTPROCESSING:')
    
    # Get autocorrelation time
    tau = sampler.get_autocorr_time(quiet=True)
    try:
        burnin = max(int(5 * np.max(tau)), 0)
        thin = max(int(0.5 * np.min(tau)), 1)
        print(f'burnin = {burnin}, thin = {thin}')
    except:
        print('WARNING thinning by 1, 0 burnin!')
        burnin = 0
        thin = 1
        
    # thin and burn
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    print('samples shape:', samples.shape)

    # get associated log probs
    samples_lnp = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)

    # format everything correctly
    samples_dict = postprocessing_get_complete_samples_dict(
        samples, samples_lnp, likelihood_manager, getRidOfFixed=getRidOfFixed, **kwargs
    )
    
    return samples_dict
