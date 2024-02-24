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

from .parameter import LogisticParameter, CartesianAngle, TrigLogisticParameter
import astropy.units as u
try:
    import gwsignal
    from gwsignal.models.teobresums import TEOBResumSDALI
    from gwsignal.core import waveform as wfm
except ImportError:
    print("Warning! Will not be able to use TEOBResumSDALI approximation")


def check_spin_settings_of_approx(approx_name):
    aligned_spins = False
    no_spins = False

    if approx_name == 'TEOBResumSDALI':
        aligned_spins = True
        return aligned_spins, no_spins

    approx = lalsim.GetApproximantFromString(approx_name)

    if not lalsim.SimInspiralImplementedTDApproximants(approx):
        raise ValueError(f"ERROR: {approx_name} is not available as a time domain waveform")

    spin_enum = lalsim.SimInspiralGetSpinSupportFromApproximant(approx)

    if spin_enum == lalsim.SIM_INSPIRAL_PRECESSINGSPIN:
        aligned_spins = False
    elif spin_enum == lalsim.SIM_INSPIRAL_ALIGNEDSPIN:
        aligned_spins = True
    elif spin_enum == lalsim.SIM_INSPIRAL_SPINLESS:
        no_spins = True
    else:
        print("WARNING, UNSURE IF WAVEFORM HAS SPINS")
    return aligned_spins, no_spins


class LogisticParameterManager:
    def __init__(self, vary_time=False, vary_skypos=False, vary_eccentricity=False, **kwargs):

        self.vary_time = vary_time
        self.vary_skypos = vary_skypos

        # get allowed spin settings of waveform model
        self.aligned_spins, self.no_spins = check_spin_settings_of_approx(kwargs['approx'])

        self.vary_eccentricity = vary_eccentricity

        # TODO put in injected values
        self.logistic_parameters = [LogisticParameter('total_mass', kwargs['mtot_lim'], None),
                                    LogisticParameter('mass_ratio', kwargs['q_lim'], None),
                                    LogisticParameter('distance_mpc', kwargs['dist_lim'], None),
                                    TrigLogisticParameter('inclination', 'cos', [-1, 1], None)
                                    ]
        if not self.no_spins:
            self.logistic_parameters.extend([
                LogisticParameter('spin1_magnitude', kwargs['chi_lim'], None),
                LogisticParameter('spin2_magnitude', kwargs['chi_lim'], None)])

        if self.vary_eccentricity:
            self.logistic_parameters.append(LogisticParameter('eccentricity', [0, 0.3], None))

        if self.vary_skypos:
            self.logistic_parameters.append(TrigLogisticParameter('declination', 'sin', [-1, 1], None))

        self.sampled_keys = [p.logistic_name for p in self.logistic_parameters]

        if not self.no_spins:
            if self.aligned_spins:
                self.sampled_keys.extend(['c1_z', 'c2_z'])
            else:
                self.sampled_keys.extend(
                    ['c1_x', 'c1_y', 'c1_z',
                     'c2_x', 'c2_y', 'c2_z'])

        self.cartesian_angles = [CartesianAngle('phase')]

        if self.vary_time:
            self.sampled_keys.append('geocenter_time')
        if self.vary_skypos:
            self.cartesian_angles.extend([
                CartesianAngle('right_ascension', phase_offset=np.pi),
                CartesianAngle('polarization')
            ])
        if self.vary_eccentricity:
            self.cartesian_angles.append(CartesianAngle('mean_anomaly'))

        for cartesian_angle in self.cartesian_angles:
            self.sampled_keys.extend([cartesian_angle.x_name, cartesian_angle.y_name])

        self.num_parameters = len(self.sampled_keys)

        self.fixed = {}
        if not self.vary_skypos:
            self.fixed['right_ascension'] = kwargs['ra']
            self.fixed['declination'] = kwargs['dec']
            self.fixed['polarization'] = kwargs['psi']

        if not self.vary_time:
            self.fixed['geocenter_time'] = kwargs['tgps_geocent']
        self.reference_time = kwargs['tgps_geocent']

        if not self.vary_eccentricity:
            self.fixed['eccentricity'] = 0
            self.fixed['mean_anomaly'] = 0

    def get_logistic_dict(self, x):
        return {self.sampled_keys[i]: x[i] for i in range(self.num_parameters)}

    def get_physical_dict_from_logistic_dict(self, x_dict):
        physicalDict = self.fixed.copy()

        physicalDict.update({transform.physical_name: transform.logistic_to_physical(x_dict[transform.logistic_name])
                             for transform in self.logistic_parameters})
        physicalDict.update({transform.physical_name: transform.cartesian_to_radian(x_dict[transform.x_name],
                                                                                    x_dict[transform.y_name])
                             for transform in self.cartesian_angles})
        return physicalDict

    def get_physical_spins(self, spin_magnitude, c_x, c_y, c_z):
        if self.no_spins:
            return 0, 0, 0

        chi_norm = spin_magnitude / np.sqrt(c_x ** 2 + c_y ** 2 + c_z ** 2)

        spin_x = c_x * chi_norm
        spin_y = c_y * chi_norm
        spin_z = c_z * chi_norm
        return spin_x, spin_y, spin_z

    def samp_to_phys(self, x):
        # Implementation of samp_to_phys
        # logistic dict
        x_dict = self.get_logistic_dict(x)
        physical_dict = self.get_physical_dict_from_logistic_dict(x_dict)

        # normalize spins
        for i in ['1', '2']:
            physical_dict[f'spin{i}_x'], physical_dict[f'spin{i}_y'], physical_dict[f'spin{i}_z'] = \
                self.get_physical_spins(physical_dict.get(f'spin{i}_magnitude', 0),
                                        x_dict.get(f'c{i}_x', 0),
                                        x_dict.get(f'c{i}_y', 0),
                                        x_dict.get(f'c{i}_z', 0))
        if self.vary_time:
            physical_dict['geocenter_time'] = x_dict['geocenter_time']

        return physical_dict

    def physical_dict_to_waveform_dict(self, physical_dict):
        """
        Take in the physical dictionary and return one that has units for astropy
        :return:
        """
        m1, m2 = m1m2_from_mtotq(physical_dict['total_mass'], physical_dict['mass_ratio'])
        return {
            'mass1': m1 * u.Msun,
            'mass2': m2 * u.Msun,
            'spin1x': physical_dict['spin1_x'] * u.dimensionless_unscaled,
            'spin1y': physical_dict['spin1_y'] * u.dimensionless_unscaled,
            'spin1z': physical_dict['spin1_z'] * u.dimensionless_unscaled,
            'spin2x': physical_dict['spin2_x'] * u.dimensionless_unscaled,
            'spin2y': physical_dict['spin2_y'] * u.dimensionless_unscaled,
            'spin2z': physical_dict['spin2_z'] * u.dimensionless_unscaled,
            'phi_ref': physical_dict['phase'] * u.rad,
            'distance': physical_dict['distance_mpc'] * u.Mpc,
            'inclination': physical_dict['inclination'] * u.rad,
            'eccentricity': physical_dict['eccentricity'] * u.dimensionless_unscaled,
            'meanPerAno': physical_dict['mean_anomaly'] * u.rad,
        }


class LnPriorManager(LogisticParameterManager):

    def initialize_walkers(self, nwalkers, injected_parameters):
        # Initialize walkers
        # (code sees unit scale quantities; use logit transformations
        # to take boundaries to +/- infinity)
        p0_arr = np.asarray([[np.random.normal() for j in range(self.num_parameters)] for i in range(nwalkers)])

        for param in self.logistic_parameters:
            p = self.sampled_keys.index(param.logistic_name)
            param_kw = param.physical_name
            # get physical parameter
            if param_kw == 'total_mass':
                param_phys = injected_parameters['mass_1'] + injected_parameters['mass_2']
            elif param_kw == 'mass_ratio':
                param_phys = injected_parameters['mass_2'] / injected_parameters['mass_1']
            elif param_kw == 'spin1_magnitude':
                param_phys = injected_parameters['a_1']
            elif param_kw == 'spin2_magnitude':
                param_phys = injected_parameters['a_2']
            elif param_kw == 'distance_mpc':
                param_phys = injected_parameters['luminosity_distance']
            else:
                try:
                    param_phys = injected_parameters[param_kw]
                    print('injected', param_phys, param_kw)
                except ValueError:
                    print(f"{param_kw} not in injected_parameters dict, continuing anyways")
                    continue
                except KeyError:
                    print(f"{param_kw} not in injected_parameters dict, continuing anyways")
                    continue

            # transform into logistic space
            param_logit = param.physical_to_logistic(param_phys)

            # draw gaussian ball in logistic space
            p0_arr[:, p] = np.asarray([np.random.normal(loc=param_logit, scale=0.05) for i in range(nwalkers)])

        # if time of coalescence sampled over need to include this separately since it isn't a unit scaled quantity
        if self.vary_time:
            index = self.sampled_keys.index('geocenter_time')
            dt_1M = 0.00127
            sigma_time = dt_1M * 2.5  # time prior from LVK has width of ~2.5M
            initial_t_walkers = np.random.normal(loc=self.reference_time, scale=sigma_time, size=nwalkers)
            p0_arr[:, index] = initial_t_walkers  # time always saved as the final param

        p0 = p0_arr.tolist()
        return p0

    def get_lnprior(self, x, **kwargs):

        x_dict = self.get_logistic_dict(x)

        # If x_phys passed in kws, return it, if not, calculate it with samp_to_phys
        phys_dict = kwargs.pop('x_phys', self.samp_to_phys(x))

        lnprior = 0
        # Logistic jacobians
        for logistic_param in self.logistic_parameters:
            lnprior += logistic_param.ln_prior_weight(phys_dict[logistic_param.physical_name])

        # cartesian angle terms
        for cartesian_param in self.cartesian_angles:
            lnprior += cartesian_param.ln_prior_weight(x_dict[cartesian_param.x_name], x_dict[cartesian_param.y_name])

        if self.vary_time:
            # gaussian
            dt_1M = 0.00127
            sigma_time = dt_1M * 2.5  # time prior from LVK has width of ~2.5M
            lnprior -= 0.5 * ((phys_dict['geocenter_time'] - self.reference_time) ** 2) / (sigma_time ** 2)

        # Spins
        if not self.no_spins:
            if self.aligned_spins:
                lnprior += -0.5 * (x_dict['c1_z'] ** 2)
                lnprior += -0.5 * (x_dict['c2_z'] ** 2)
            else:
                lnprior += -0.5 * (x_dict['c1_x'] ** 2 + x_dict['c1_y'] ** 2 + x_dict['c1_z'] ** 2)
                lnprior += -0.5 * (x_dict['c2_x'] ** 2 + x_dict['c2_y'] ** 2 + x_dict['c2_z'] ** 2)

        return lnprior


class WaveformManager(LogisticParameterManager):
    def __init__(self, *args, **kwargs):
        super(WaveformManager, self).__init__(*args, **kwargs)
        self.approx_name = kwargs['approx']
        self.approximant = lalsim.SimInspiralGetApproximantFromString(self.approx_name)

        return

    def generate_lal_hphc(self, m1_msun, m2_msun, chi1, chi2, dist_mpc=1,
                          dt=None, f_low=20, f_ref=11, inclination=0, phi_ref=0., eccentricity=0,
                          mean_anomaly_periastron=0):
        """
        Generate the plus and cross polarizations for given waveform parameters and approximant
        """

        m1_kg = m1_msun * lal.MSUN_SI
        m2_kg = m2_msun * lal.MSUN_SI

        distance = dist_mpc * 1e6 * lal.PC_SI

        param_dict = lal.CreateDict()

        hp, hc = lalsim.SimInspiralChooseTDWaveform(m1_kg, m2_kg,
                                                    chi1[0], chi1[1], chi1[2],
                                                    chi2[0], chi2[1], chi2[2],
                                                    distance, inclination,
                                                    phi_ref, 0., eccentricity, mean_anomaly_periastron,
                                                    dt, f_low, f_ref,
                                                    param_dict,
                                                    self.approximant)
        return hp, hc

    def get_hplus_hcross(self, x_phys, f_low=11, f_ref=11, delta_t=None):
        """
        get complex waveform at geocenter
        """
        m1, m2 = m1m2_from_mtotq(x_phys['total_mass'], x_phys['mass_ratio'])
        chi1 = [x_phys['spin1_x'], x_phys['spin1_y'], x_phys['spin1_z']]
        chi2 = [x_phys['spin2_x'], x_phys['spin2_y'], x_phys['spin2_z']]

        hp, hc = self.generate_lal_hphc(m1, m2, chi1, chi2,
                                       dist_mpc=x_phys['distance_mpc'], dt=delta_t,
                                       f_low=f_low, f_ref=f_ref,
                                       inclination=x_phys['inclination'],
                                       phi_ref=x_phys['phase'],
                                       eccentricity=x_phys['eccentricity'],
                                       mean_anomaly_periastron=x_phys['mean_anomaly'])
        return hp, hc

    def get_projected_waveform(self, x_phys, ifos, time_dict, f_low=11, f_ref=11,
                               delta_t=None,
                               ap_dict=None, tpeak_dict=None):

        hp, hc = self.get_hplus_hcross(x_phys, f_low=f_low, f_ref=f_ref, delta_t=delta_t)

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
        projected_waveform_dict = {}
        for ifo in ifos:
            # Antenna patterns and tpeak
            Fp, Fc = AP_dict[ifo]
            tt = TP_dict[ifo]  # triggertime = peak time, NOT tcut (cutoff time)

            # Generate waveform
            h = rwf.generate_lal_waveform(hplus=hp, hcross=hc,
                                          times=time_dict[ifo],
                                          triggertime=tt)

            # Project onto detector
            h_ifo = Fp * h.real - Fc * h.imag

            projected_waveform_dict[ifo] = h_ifo

        return projected_waveform_dict


class NewWaveformManager(LogisticParameterManager):
    def __init__(self, *args, **kwargs):
        super(NewWaveformManager, self).__init__(*args, **kwargs)

        self.approx_name = kwargs['approx']
        if self.approx_name == 'TEOBResumSDALI':
            self.generator = TEOBResumSDALI(modes_to_use=[[2, 2]])
        else:
            self.generator = gwsignal.core.waveform.LALCompactBinaryCoalescenceGenerator(self.approx_name)
        return

    def get_hplus_hcross(self, x_phys, f_low=11, f_ref=11, delta_t=None):
        """
        get complex waveform at geocenter
        """
        params = self.physical_dict_to_waveform_dict(x_phys)
        params['f22_start'] = f_low * u.Hz
        params['deltaT'] = delta_t * u.s
        params['f22_ref'] = f_ref * u.Hz
        params['condition'] = 0

        return wfm.GenerateTDWaveform(params, self.generator)

    def get_projected_waveform(self, x_phys, ifos, time_dict, f_low=11, f_ref=11,
                               delta_t=None,
                               ap_dict=None, tpeak_dict=None):

        hp, hc = self.get_hplus_hcross(x_phys, f_low=f_low, f_ref=f_ref, delta_t=delta_t)

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
        projected_waveform_dict = {}
        for ifo in ifos:
            # Antenna patterns and tpeak
            Fp, Fc = AP_dict[ifo]
            tt = TP_dict[ifo]  # triggertime = peak time, NOT tcut (cutoff time)

            # Generate waveform
            h = rwf.generate_lal_waveform(hplus=hp, hcross=hc,
                                          times=time_dict[ifo],
                                          triggertime=tt)

            # Project onto detector
            h_ifo = Fp * h.real - Fc * h.imag

            projected_waveform_dict[ifo] = h_ifo

        return projected_waveform_dict


class LnLikelihoodManager(LogisticParameterManager):
    def __init__(self, *args, **kwargs):
        try:
            self.waveform_manager = NewWaveformManager(*args, **kwargs)
        except Exception as e:
            print(e)
            print("warning, new waveform manager has failed to be created, using old waveform manager")
            self.waveform_manager = WaveformManager(*args, **kwargs)
        self.log_prior = LnPriorManager(*args, **kwargs)

        super().__init__(*args, **kwargs)

    def get_lnprob(self, x, f_low=11, f_ref=11, return_wf=False,
                   only_prior=False,
                   rho_dict=None, time_dict=None, delta_t=None, data_dict=None,
                   ap_dict=None, tpeak_dict=None, **kwargs):
        # get physical parameters
        x_phys = self.samp_to_phys(x)

        # Initialize posterior to 0
        lnprob = 0

        # Calculate posterior
        if not only_prior:
            projected_wf_dict = self.waveform_manager.get_projected_waveform(
                x_phys, data_dict.keys(), time_dict, f_low=f_low, f_ref=f_ref,
                delta_t=delta_t,
                ap_dict=ap_dict, tpeak_dict=tpeak_dict
            )

            # Cycle through ifos
            for ifo, data in data_dict.items():

                # for debugging purporses
                if return_wf == ifo:
                    return projected_wf_dict[ifo]

                # Truncate and compute residuals
                r = data - projected_wf_dict[ifo]

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