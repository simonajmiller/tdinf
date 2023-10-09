from pylab import *
from scipy.linalg import solve_toeplitz
import lal
import lalsimulation as lalsim
try: 
    import reconstructwf as rwf
    import io
    from spins_and_masses import m1m2_from_mtotq
except: 
    from . import reconstructwf as rwf
    from . import io
    from .spins_and_masses import m1m2_from_mtotq

def logit(x, xmin=0, xmax=1):
    return log(x - xmin) - log(xmax - x)

def inv_logit(y, xmin=0, xmax=1):
    return (exp(y)*xmax + xmin)/(1 + exp(y))

def logit_jacobian(x, xmin=0, xmax=1):
    return 1./(x-xmin) + 1./(xmax-x)

def samp_to_phys(x, **kws):
    
    if len(x) == 14:
        # default
        x_mtot, x_q, x_chi1, x_chi2, x_dist_mpc, phi_x, phi_y, c1x, c1y, c1z, c2x, c2y, c2z, x_cosi = x
    elif len(x) == 15:
        # add time
        x_mtot, x_q, x_chi1, x_chi2, x_dist_mpc, phi_x, phi_y, c1x, c1y, c1z, c2x, c2y, c2z, x_cosi, t = x
    elif len(x) == 19:
        # add polarization + skyloc 
        x_mtot, x_q, x_chi1, x_chi2, x_dist_mpc, phi_x, phi_y, c1x, c1y, c1z, c2x, c2y, c2z, x_cosi, ra_x, ra_y, x_sindec, psi_x, psi_y = x
    elif len(x) == 20:
        # add polarization + skyloc + time
        x_mtot, x_q, x_chi1, x_chi2, x_dist_mpc, phi_x, phi_y, c1x, c1y, c1z, c2x, c2y, c2z, x_cosi, ra_x, ra_y, x_sindec, psi_x, psi_y, t = x
    else:
        print(f'length of x not valid: len(x) = {len(x)}')
        sys.exit()
        
    # undo logit transformations
    mtot = inv_logit(x_mtot, kws['mtot_lim'][0], kws['mtot_lim'][1])
    q = inv_logit(x_q, kws['q_lim'][0], kws['q_lim'][1])
    chi1 = inv_logit(x_chi1, kws['chi_lim'][0], kws['chi_lim'][1])
    chi2 = inv_logit(x_chi2, kws['chi_lim'][0], kws['chi_lim'][1])
    dist_mpc = inv_logit(x_dist_mpc, kws['dist_lim'][0], kws['dist_lim'][1])
    iota = np.arccos(inv_logit(x_cosi, -1, 1))
        
    # deal with time 
    if len(x) == 15 or len(x)==20: 
        # don't need to sample time in logit space
        tgps_geocent = t
    else: 
        tgps_geocent= kws['tgps_geocent']
    
    # deal with polarization, skyposition,
    if len(x) >= 19: 
        # get ra from quadratures
        ra = arctan2(ra_y, ra_x) + np.pi
        # get dec from sin dec
        dec = np.arcsin(inv_logit(x_sindec, -1, 1))
        # get polarization from quadratures
        psi = arctan2(psi_y, psi_x)
    else: 
        ra = kws['ra']
        dec = kws['dec']
        psi = kws['psi']
    
    # renormalize 3D spins
    chi1_norm = chi1 / sqrt(c1x**2 + c1y**2 + c1z**2)
    chi1x = c1x * chi1_norm
    chi1y = c1y * chi1_norm
    chi1z = c1z * chi1_norm

    chi2_norm = chi2 / sqrt(c2x**2 + c2y**2 + c2z**2)
    chi2x = c2x * chi2_norm
    chi2y = c2y * chi2_norm
    chi2z = c2z * chi2_norm

    # get phase from quadratures
    phi_ref = arctan2(phi_y, phi_x)
    
    return mtot, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, dist_mpc, phi_ref, iota, ra, dec, psi, tgps_geocent


def samp_to_phys_angles(x, **kwargs):
        
    # get physical parameters
    x_phys = np.array([samp_to_phys(x_i, **kwargs) for x_i in x.T], ndmin=2)
                
    # change to LALInference spin convention
    ys = []
    for x_i in x_phys:
        mtot, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, dist_mpc, phi_ref, iota, ra, dec, psi, tgps_geocent = x_i
        m1, m2 = m1m2_from_mtotq(mtot, q)    
        ys.append(lalsim.SimInspiralTransformPrecessingWvf2PE(
            iota, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, m1, m2,
            kwargs['f_low'], phi_ref
        ))
    # quantities to return
    theta_jn, phi_jl, tilt1, tilt2, phi12, chi1, chi2 = np.array(ys, ndmin=2).T
    mtot, q, _, _, _, _, _, _, dist_mpc, phi_ref, iota, ra, dec, psi, tgps_geocent = np.array(x_phys.T)
    return mtot, q, chi1, chi2, phi_jl, tilt1, tilt2, phi12, dist_mpc, phi_ref, theta_jn, iota, ra, dec, psi, tgps_geocent 


def get_dict_from_samples(samples, angles=False, **kwargs):    
    keys = ['mtotal', 'q', 'chi1', 'chi2', 'phi_jl', 'tilt1', 'tilt2', 'phi12', 'dist', 'phase', 'theta_jn', 'iota',
                'ra', 'dec', 'psi', 'tgps_geocent']
    f = samp_to_phys_angles
    return dict(zip(keys, f(samples.T, **kwargs)))

'''
Prior function
'''
def get_lnprior(x, **kws):
    
    if len(x) == 14:
        # default
        x_mtot, x_q, x_chi1, x_chi2, x_dist_mpc, phi_x, phi_y, c1x, c1y, c1z, c2x, c2y, c2z, x_cosi = x
    elif len(x) == 15:
        # add time
        x_mtot, x_q, x_chi1, x_chi2, x_dist_mpc, phi_x, phi_y, c1x, c1y, c1z, c2x, c2y, c2z, x_cosi, t = x
    elif len(x) == 19:
        # add polarization + skyloc 
        x_mtot, x_q, x_chi1, x_chi2, x_dist_mpc, phi_x, phi_y, c1x, c1y, c1z, c2x, c2y, c2z, x_cosi, ra_x, ra_y, x_sindec, psi_x, psi_y = x
    elif len(x) == 20:
        # add polarization + skyloc + time
        x_mtot, x_q, x_chi1, x_chi2, x_dist_mpc, phi_x, phi_y, c1x, c1y, c1z, c2x, c2y, c2z, x_cosi, ra_x, ra_y, x_sindec, psi_x, psi_y, t = x
    else:
        print(f'length of x not valid: len(x) = {len(x)}')
        sys.exit()
    
    # If x_phys passed in kws, return it, if not, calculate it with samp_to_phys
    x_phys = kws.pop('x_phys', samp_to_phys(x, **kws))
    mtot, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, dist_mpc, phi_ref, iota, ra, dec, psi, tgps_geocent = x_phys
    
    # Spin magnitudes
    chi1 = sqrt(chi1x**2 + chi1y**2 + chi1z**2)
    chi2 = sqrt(chi2x**2 + chi2y**2 + chi2z**2)
    
    # Gaussian prior for phase quadratures
    lnprior = -0.5*(phi_x**2 + phi_y**2)
        
    # Logistic jacobians
    for k, v in zip(['mtot', 'q', 'chi', 'chi', 'dist'],
                    [mtot, q, chi1, chi2, dist_mpc]):
        v_min, v_max = kws['%s_lim' % k]
        lnprior -= log(logit_jacobian(v, v_min, v_max))
        
    if len(x) >= 14: # if sampling incl
        lnprior -= log(logit_jacobian(cos(iota), -1, 1))
    
    if len(x) >= 19: # if sampling pol + skypos
        lnprior -= 0.5*(psi_x**2 + psi_y**2)
        lnprior -= 0.5*(ra_x**2 + ra_y**2)
        lnprior -= log(logit_jacobian(sin(dec), -1, 1))
        
    if len(x) == 15 or len(x)==20: # if sampling time 
        # gaussian 
        dt_1M = 0.00127
        sigma_time = dt_1M*2.5 # time prior from LVK has width of ~2.5M
        lnprior -= 0.5*((t-kws['tgps_geocent'])**2)/(sigma_time**2)
        
    # Spins
    lnprior += -0.5*(c1x**2 + c1y**2 + c1z**2)
    lnprior += -0.5*(c2x**2 + c2y**2 + c2z**2)
        
    return lnprior

'''
Posterior function
'''
def get_lnprob(x, f_low=11, f_ref=11, return_wf=False, return_params=False, 
               only_prior=False, approx='NRSur7dq4',
               rho_dict=None, time_dict=None, delta_t=None, data_dict=None,
               ap_dict=None, tpeak_dict=None, **kwargs):
    
    # get physical parameters
    x_phys = samp_to_phys(x, **kwargs)
    
    # Intialize posterior to 0
    lnprob = 0
    
    # Calculate posterior
    if not only_prior:
        
        # get complex-valued waveform at geocenter
        mtot, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, dist_mpc, phi_ref, iota, ra, dec, psi, tgps_geocent = x_phys
        
        m1, m2 = m1m2_from_mtotq(mtot, q)
        chi1 = [chi1x, chi1y, chi1z]
        chi2 = [chi2x, chi2y, chi2z]

        hp, hc = rwf.generate_lal_hphc(approx, m1, m2, chi1, chi2,
                                       dist_mpc=dist_mpc, dt=delta_t,
                                       f_low=f_low, f_ref=f_ref,
                                       inclination=iota, phi_ref=phi_ref) 
        
        # Which interferometers are we sampling over?
        ifos = data_dict.keys()

        # If we are sampling over sky position and/or time ...
        if ap_dict is None and tpeak_dict is None: # both
            TP_dict, AP_dict = io.get_tgps_and_ap_dicts(tgps_geocent, ifos, ra, dec, psi, verbose=False) 
        elif ap_dict is None: # just skypos
            _, AP_dict = io.get_tgps_and_ap_dicts(tgps_geocent, ifos, ra, dec, psi, verbose=False) 
            TP_dict = tpeak_dict.copy()
        elif tpeak_dict is None: # just time
            TP_dict, _ = io.get_tgps_and_ap_dicts(tgps_geocent, ifos, ra, dec, psi, verbose=False) 
            AP_dict = ap_dict.copy()
        else: # neither
            TP_dict = tpeak_dict.copy()
            AP_dict = ap_dict.copy()
                
        # Cycle through ifos
        for ifo, data in data_dict.items():
                        
            # Antenna patterns and tpeak 
            Fp, Fc = AP_dict[ifo]
            tt = TP_dict[ifo] # triggertime = peak time, NOT tcut (cutoff time)
                
            # Generate waveform
            h = rwf.generate_lal_waveform(hplus=hp, hcross=hc,
                                          times=time_dict[ifo], 
                                          triggertime=tt) 
            
            # Project onto detector
            h_ifo = Fp*h.real - Fc*h.imag

            # for debugging purporses
            if return_wf == ifo:
                return h_ifo

            # Truncate and compute residuals
            r = data - h_ifo

            # "Over whiten" residuals
            rwt = solve_toeplitz(rho_dict[ifo], r)

            # Compute log likelihood for ifo
            lnprob -= 0.5*np.dot(r, rwt)
        
    # Calculate prior
    lnprob += get_lnprior(x, x_phys=x_phys, **kwargs)
    
    # Check for NaN
    if lnprob!=lnprob: 
        params = [mtot, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, dist_mpc, phi_ref, iota, ra, dec, psi, tgps_geocent]
        print('lnprob = NaN for:') 
        print(f'mtot, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, dist_mpc, phi_ref, iota, ra, dec, psi, tgps_geocent = {params}')
        return -np.inf
    
    # Return posterior
    return lnprob
