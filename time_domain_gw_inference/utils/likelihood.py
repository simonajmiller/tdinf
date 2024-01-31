import numpy as np
from scipy.linalg import solve_toeplitz
import lal
import lalsimulation as lalsim
import sys

try:
    import reconstructwf as rwf
    from spins_and_masses import m1m2_from_mtotq
    from misc import logit, inv_logit, logit_jacobian
except:
    from . import reconstructwf as rwf
    from .spins_and_masses import m1m2_from_mtotq
    from .misc import logit, inv_logit, logit_jacobian


class LogisticParameterManager:
    def __init__(self, vary_time=False, vary_skypos=False, **kwargs):

        self.vary_time = vary_time
        self.vary_skypos = vary_skypos

        self.logistic_keys = ['x_total_mass', 'x_mass_ratio',
                              'x_spin1_magnitude', 'x_spin2_magnitude',
                              'x_distance_mpc',
                              'phase_x', 'phase_y',
                              'c1_x', 'c1_y', 'c1_z',
                              'c2_x', 'c2_y', 'c2_z',
                              'x_cos_inclination']
        if self.vary_time:
            self.logistic_keys.append('geocenter_time')
        if self.vary_skypos:
            self.logistic_keys.extend(['right_ascension_x', 'right_ascension_y',
                                       'x_sin_declination',
                                       'polarization_x', 'polarization_y'])

        self.num_parameters = len(self.logistic_keys)

        self.limit = {
            'total_mass': kwargs['mtot_lim'],
            'mass_ratio': kwargs['q_lim'],
            'spin1_magnitude': kwargs['chi_lim'],
            'spin2_magnitude': kwargs['chi_lim'],
            'distance_mpc': kwargs['dist_lim'],
            'cos_inclination': [-1, 1],
        }

        self.fixed = {}
        if not self.vary_skypos:
            self.fixed['right_ascension'] = kwargs['ra']
            self.fixed['declination'] = kwargs['dec']
            self.fixed['polarization'] = kwargs['psi']
        else:
            self.limit['sin_declination'] = [-1, 1]

        if not self.vary_time:
            self.fixed['geocenter_time'] = kwargs['tgps_geocent']
        self.reference_time = kwargs['tgps_geocent']

    def get_logistic_dict(self, x):
        return {self.logistic_keys[i]: x[i] for i in range(self.num_parameters)}

    def get_physical_dict_from_intrinsic_parameters(self, x_dict):
        # undo logit transformations
        return {
            'total_mass': inv_logit(x_dict['x_total_mass'], *self.limit['total_mass']),
            'mass_ratio': inv_logit(x_dict['x_mass_ratio'], *self.limit['mass_ratio']),
            'spin1_magnitude': inv_logit(x_dict['x_spin1_magnitude'], *self.limit['spin1_magnitude']),
            'spin2_magnitude': inv_logit(x_dict['x_spin2_magnitude'], *self.limit['spin2_magnitude']),
            'distance_mpc': inv_logit(x_dict['x_distance_mpc'], *self.limit['distance_mpc']),
            'inclination': np.arccos(inv_logit(x_dict['x_cos_inclination'], *self.limit['cos_inclination'])),
            'phase': np.arctan2(x_dict['phase_y'], x_dict['phase_x'])
        }

    def get_physical_spins(self, spin_magnitude, c_x, c_y, c_z):
        chi_norm = spin_magnitude / np.sqrt(c_x ** 2 + c_y ** 2 + c_z ** 2)
        spin_x = c_x * chi_norm
        spin_y = c_y * chi_norm
        spin_z = c_z * chi_norm
        return spin_x, spin_y, spin_z

    def get_physical_extrinsic(self, x_dict):
        if self.vary_skypos:
            # get ra from quadratures
            ra = np.arctan2(x_dict['right_ascension_y'], x_dict['right_ascension_x']) + np.pi
            # get dec from sin dec
            dec = np.arcsin(inv_logit(x_dict['x_sin_declination'], *self.limit['sin_declination']))
            # get polarization from quadratures
            psi = np.arctan2(x_dict['polarization_y'], x_dict['polarization_x'])
        else:
            ra = self.fixed['right_ascension']
            dec = self.fixed['declination']
            psi = self.fixed['polarization']

        return {
            'right_ascension': ra, 'declination': dec, 'polarization': psi
        }

    def samp_to_phys(self, x):
        # Implementation of samp_to_phys
        # logistic dict
        x_dict = self.get_logistic_dict(x)

        physical_dict = self.get_physical_dict_from_intrinsic_parameters(x_dict)

        # normalize spins
        for i in ['1', '2']:
            physical_dict[f'spin{i}_x'], physical_dict[f'spin{i}_y'], physical_dict[f'spin{i}_z'] = \
                self.get_physical_spins(physical_dict[f'spin{i}_magnitude'],
                                        x_dict[f'c{i}_x'], x_dict[f'c{i}_y'], x_dict[f'c{i}_z'])

        if self.vary_time:
            physical_dict['geocenter_time'] = x_dict['geocenter_time']
        else:
            physical_dict['geocenter_time'] = self.fixed['geocenter_time']

        physical_dict.update(self.get_physical_extrinsic(x_dict))
        return physical_dict


class LnPriorManager(LogisticParameterManager):

    def get_lnprior(self, x, **kwargs):

        x_dict = self.get_logistic_dict(x)

        # If x_phys passed in kws, return it, if not, calculate it with samp_to_phys
        phys_dict = kwargs.pop('x_phys', self.samp_to_phys(x))

        # Gaussian prior for phase quadratures
        lnprior = -0.5 * (x_dict['phase_x'] ** 2 + x_dict['phase_y'] ** 2)

        # Logistic jacobians
        for key in ['total_mass', 'mass_ratio', 'spin1_magnitude', 'spin2_magnitude', 'distance_mpc']:
            lnprior -= np.log(logit_jacobian(phys_dict[key], *self.limit[key]))

        lnprior -= np.log(logit_jacobian(np.cos(phys_dict['inclination']),
                                         *self.limit['cos_inclination']))

        if self.vary_skypos:
            lnprior -= 0.5 * (x_dict['polarization_x'] ** 2 + x_dict['polarization_y'] ** 2)
            lnprior -= 0.5 * (x_dict['right_ascension_x'] ** 2 + x_dict['right_ascension_y'] ** 2)
            lnprior -= np.log(logit_jacobian(np.sin(phys_dict['declination']), *self.limit['sin_declination']))

        if self.vary_time:
            # gaussian
            dt_1M = 0.00127
            sigma_time = dt_1M * 2.5  # time prior from LVK has width of ~2.5M
            lnprior -= 0.5 * ((phys_dict['geocenter_time'] - self.reference_time) ** 2) / (sigma_time ** 2)

        # Spins
        lnprior += -0.5 * (x_dict['c1_x'] ** 2 + x_dict['c1_y'] ** 2 + x_dict['c1_z'] ** 2)
        lnprior += -0.5 * (x_dict['c2_x'] ** 2 + x_dict['c2_y'] ** 2 + x_dict['c2_z'] ** 2)

        return lnprior


class LnLikelihoodManager(LogisticParameterManager):
    def __init__(self, vary_time=False, vary_skypos=False, **kwargs):
        self.log_prior = LnPriorManager(vary_time=vary_time, vary_skypos=vary_skypos, **kwargs)

        super().__init__(vary_time=vary_time, vary_skypos=vary_skypos, **kwargs)

    def get_lnprob(self, x, f_low=11, f_ref=11, return_wf=False, return_params=False,
                   only_prior=False, approx='NRSur7dq4',
                   rho_dict=None, time_dict=None, delta_t=None, data_dict=None,
                   ap_dict=None, tpeak_dict=None, **kwargs):
        # get physical parameters
        x_phys = self.samp_to_phys(x)

        # Intialize posterior to 0
        lnprob = 0

        # Calculate posterior
        if not only_prior:

            # get complex-valued waveform at geocenter
            m1, m2 = m1m2_from_mtotq(x_phys['total_mass'], x_phys['mass_ratio'])
            chi1 = [x_phys['spin1_x'], x_phys['spin1_y'], x_phys['spin1_z']]
            chi2 = [x_phys['spin2_x'], x_phys['spin2_y'], x_phys['spin2_z']]

            hp, hc = rwf.generate_lal_hphc(approx, m1, m2, chi1, chi2,
                                           dist_mpc=x_phys['distance_mpc'], dt=delta_t,
                                           f_low=f_low, f_ref=f_ref,
                                           inclination=x_phys['inclination'],
                                           phi_ref=x_phys['phase'])

            # Which interferometers are we sampling over?
            ifos = data_dict.keys()

            # If we are sampling over sky position and/or time ...
            if self.vary_skypos and self.vary_time:
                TP_dict, AP_dict = rwf.get_tgps_and_ap_dicts(
                    x_phys['geocenter_time'],
                    ifos,
                    x_phys['right_ascension'], x_phys['declination'], x_phys['polarization'],
                    verbose=False)
            elif self.vary_skypos:  # just skypos
                _, AP_dict = rwf.get_tgps_and_ap_dicts(
                    self.reference_time, ifos,
                    x_phys['right_ascension'], x_phys['declination'], x_phys['polarization'],
                    verbose=False)
                TP_dict = tpeak_dict.copy()
            elif tpeak_dict is None:  # just time
                TP_dict, _ = rwf.get_tgps_and_ap_dicts(
                    x_phys['geocenter_time'], ifos,
                    self.fixed['right_ascension'], self.fixed['declination'], self.fixed['polarization'],
                    verbose=False)
                AP_dict = ap_dict.copy()
            else:  # neither
                TP_dict = tpeak_dict.copy()
                AP_dict = ap_dict.copy()

            # Cycle through ifos
            for ifo, data in data_dict.items():

                # Antenna patterns and tpeak
                Fp, Fc = AP_dict[ifo]
                tt = TP_dict[ifo]  # triggertime = peak time, NOT tcut (cutoff time)

                # Generate waveform
                h = rwf.generate_lal_waveform(hplus=hp, hcross=hc,
                                              times=time_dict[ifo],
                                              triggertime=tt)

                # Project onto detector
                h_ifo = Fp * h.real - Fc * h.imag

                # for debugging purporses
                if return_wf == ifo:
                    return h_ifo

                # Truncate and compute residuals
                r = data - h_ifo

                # "Over whiten" residuals
                rwt = solve_toeplitz(rho_dict[ifo], r)

                # Compute log likelihood for ifo
                lnprob -= 0.5 * np.dot(r, rwt)

        # Calculate prior
        lnprob += self.log_prior.get_lnprior(x, x_phys=x_phys, **kwargs)

        # Check for NaN
        if lnprob != lnprob:
            print('lnprob = NaN for:')
            print(x_phys)
            return -np.inf

        # Return posterior
        return lnprob