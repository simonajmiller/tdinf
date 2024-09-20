import numpy as np
from scipy.signal import tukey
from scipy.linalg import solve_toeplitz
from scipy.interpolate import interp1d
import lal
import lalsimulation as lalsim
import sys
from gwpy.timeseries import TimeSeries

# try:
#     import reconstructwf as rwf
#     from spins_and_masses import m1m2_from_mtotq
#     from misc import logit, inv_logit, logit_jacobian
# except:
from . import reconstructwf as rwf
from .spins_and_masses import m1m2_from_mtotq
from .misc import logit, inv_logit, logit_jacobian, calc_mf_SNR, calc_opt_SNR, calc_network_SNR

from .parameter import LogisticParameter, CartesianAngle, TrigLogisticParameter
from .preprocessing import get_ACF

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

def interpolate_timeseries(time, values, new_time_grid):
    """
    Interpolates a timeseries to a new grid of points using cubic interpolation

    Parameters:
    - time: array-like, the original time grid
    - values: array-like, the values of the timeseries at the original time grid
    - new_time_grid: array-like, the new time grid

    Returns:
    - value_on_grid: array-like, the interpolated values at the new time grid
    """
#     # linear interp:
#     value_on_grid = np.interp(new_time_grid, time, values, left=0, right=0) 
    
    # cubic interp:
    values_interpolator = interp1d(time, values, kind='cubic', fill_value=0, bounds_error=False)
    value_on_grid = values_interpolator(new_time_grid)
    
    return value_on_grid

def apply_window(timeseries): 
    nsamps = len(timeseries)
    window = tukey(nsamps)
    window[int(0.5*nsamps):] = 1.
    return timeseries*window

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
                                    LogisticParameter('luminosity_distance', kwargs['dist_lim'], None),
                                    TrigLogisticParameter('inclination', 'cos', [-1, 1], None)
                                    ]
        if not self.no_spins:
            self.logistic_parameters.extend([
                LogisticParameter('spin1_magnitude', kwargs['chi_lim'], None),
                LogisticParameter('spin2_magnitude', kwargs['chi_lim'], None)])

        if self.vary_eccentricity:
            self.logistic_parameters.append(LogisticParameter('eccentricity', kwargs['eccentricity_lim'], None))

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
            # this is set for the prior manager
            self.sigma_time = kwargs['sigma_time']
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
            self.fixed['right_ascension'] = kwargs['right_ascension']
            self.fixed['declination'] = kwargs['declination']
            self.fixed['polarization'] = kwargs['polarization']

        if not self.vary_time:
            self.fixed['geocenter_time'] = kwargs['geocenter_time']
        self.reference_time = kwargs['geocenter_time']

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

    @staticmethod
    def physical_dict_to_waveform_dict(physical_dict):
        """
        Take in the physical dictionary and return one that has units for astropy
        :return:
        """
        m1, m2 = m1m2_from_mtotq(physical_dict['total_mass'], physical_dict['mass_ratio'])
        param_dict = {
            'mass1': m1 * u.Msun,
            'mass2': m2 * u.Msun,
            'spin1x': physical_dict['spin1_x'] * u.dimensionless_unscaled,
            'spin1y': physical_dict['spin1_y'] * u.dimensionless_unscaled,
            'spin1z': physical_dict['spin1_z'] * u.dimensionless_unscaled,
            'spin2x': physical_dict['spin2_x'] * u.dimensionless_unscaled,
            'spin2y': physical_dict['spin2_y'] * u.dimensionless_unscaled,
            'spin2z': physical_dict['spin2_z'] * u.dimensionless_unscaled,
            'phi_ref': physical_dict['phase'] * u.rad,
            'distance': physical_dict['luminosity_distance'] * u.Mpc,
            'inclination': physical_dict['inclination'] * u.rad,
        }
        # TODO, maybe it should check if we have an eccentric waveform?
        if 'eccentricity' in physical_dict:
            param_dict['eccentricity'] = physical_dict['eccentricity'] * u.dimensionless_unscaled
            param_dict['meanPerAno'] = physical_dict['mean_anomaly'] * u.rad
        return param_dict


class LnPriorManager(LogisticParameterManager):

    def initialize_walkers(self, nwalkers, injected_parameters, reference_posteriors=None, verbose=False):

        logistic_params = self.logistic_parameters
        cartesian_params = self.cartesian_angles

        # Initialize walkers randomly
        p0_arr = np.asarray([[np.random.normal() for j in range(self.num_parameters)] for i in range(nwalkers)])

        if reference_posteriors is not None:
            idxs = np.random.choice(range(len(reference_posteriors['right_ascension'])), size=nwalkers)

        for param in logistic_params + cartesian_params:
            param_kw = param.physical_name
            param_logistic = isinstance(param, LogisticParameter) or isinstance(param, TrigLogisticParameter)
            p = self.sampled_keys.index(param.logistic_name if param_logistic else f"{param_kw}_x")

            # Flag to decide whether to use reference posterior or injected parameters
            use_reference_posterior = False

            if reference_posteriors is not None and param_kw in reference_posteriors:
                walkers_phys = reference_posteriors[param_kw].copy()
                # Check if reference posterior is constant (e.g., zero spread)
                if not np.allclose(walkers_phys, walkers_phys[0]):
                    use_reference_posterior = True  # Only use reference posterior if there's sufficient spread
                    # there are no prior bounds for cartesian angles, so don't check them
                    if not param_logistic:
                        break
                    # check if the reference posterior is outisde the prior
                    for i, param_phys in enumerate(walkers_phys):

                        if param_phys < param.limit[0] or param_phys > param.limit[1]:
                            new_param_phys = np.random.uniform(param.limit[0], param.limit[1], size=1)[0]
                            print(f"WARNING: Injected value ({param_phys}) for {param_kw} is outside limit {param.limit} for value in posterior, setting it to {new_param_phys}")
                            reference_posteriors.at[i, param_kw] = new_param_phys

            if use_reference_posterior:
                walkers_phys = reference_posteriors[param_kw][idxs]

                if type(param) == TrigLogisticParameter:
                    walkers_phys = param.trig_function(walkers_phys)

                if param_logistic:
                    # Logistic parameter transformation
                    walkers_logit = param.physical_to_logistic(walkers_phys)
                    p0_arr[:, p] = walkers_logit
                else:
                    # Cartesian parameter transformation
                    initial_x, initial_y = param.radian_to_cartesian(walkers_phys)
                    p0_arr[:, p] = -1.0 * initial_x
                    p0_arr[:, self.sampled_keys.index(f"{param_kw}_y")] = -1.0 * initial_y

            else:
                # Use injected parameter as fallback
                try:
                    param_phys = injected_parameters[param_kw]
                    # If given (e.g. declination, get sin declination like we need)
                    if type(param) == TrigLogisticParameter:
                        param_phys = param.trig_function(param_phys)
                    if verbose:
                        print('injected', param_phys, param_kw)
                except (ValueError, KeyError):
                    if verbose:
                        print(f"{param_kw} not in injected_parameters dict, continuing anyways")
                    continue

                # only need to check boundaries for logistic params, not for cartesian angles
                if param_logistic:
                    # Check for boundary conditions
                    if param_phys in param.limit:
                        print(f'reference value of {param_kw} is on boundary of acceptable range, '
                              f'drawing random value from within range, avoiding infinity!')
                        continue
                    elif param_phys <= param.limit[0] or param_phys >= param.limit[1]:
                        print(f"WARNING: Injected value ({param_phys}) for {param_kw} is outside limit {param.limit}. "
                              f"Will not set initial values around this value")
                        continue

                    param_logit = param.physical_to_logistic(param_phys)
                    p0_arr[:, p] = np.random.normal(loc=param_logit, scale=0.05, size=nwalkers)

        # Handle time separately if varying it
        if self.vary_time:
            if reference_posteriors is not None and 'geocenter_time' in reference_posteriors:
                p0_arr[:, self.sampled_keys.index('geocenter_time')] = reference_posteriors['geocenter_time'][idxs]
            else:
                initial_t_walkers = np.random.normal(loc=self.reference_time, scale=self.sigma_time, size=nwalkers)
                p0_arr[:, self.sampled_keys.index('geocenter_time')] = initial_t_walkers

        p0 = p0_arr.tolist()
        return p0

    def get_lnprior(self, x, phys_dict=None):

        x_dict = self.get_logistic_dict(x)

        # If phys_dict passed in kws, return it, if not, calculate it with samp_to_phys
        if phys_dict is None:
            phys_dict = self.samp_to_phys(x)

        lnprior = 0
        # Logistic jacobians
        for logistic_param in self.logistic_parameters:
            lnprior += logistic_param.ln_prior_weight(phys_dict[logistic_param.physical_name])

        # cartesian angle terms
        for cartesian_param in self.cartesian_angles:
            lnprior += cartesian_param.ln_prior_weight(x_dict[cartesian_param.x_name], x_dict[cartesian_param.y_name])

        if self.vary_time:
            # gaussian
            lnprior -= 0.5 * ((phys_dict['geocenter_time'] - self.reference_time) ** 2) / (self.sigma_time ** 2)

        # Spins
        if not self.no_spins:
            if self.aligned_spins:
                lnprior += -0.5 * (x_dict['c1_z'] ** 2)
                lnprior += -0.5 * (x_dict['c2_z'] ** 2)
            else:
                lnprior += -0.5 * (x_dict['c1_x'] ** 2 + x_dict['c1_y'] ** 2 + x_dict['c1_z'] ** 2)
                lnprior += -0.5 * (x_dict['c2_x'] ** 2 + x_dict['c2_y'] ** 2 + x_dict['c2_z'] ** 2)

        return lnprior


class AntennaAndTimeManager(LogisticParameterManager):
    def __init__(self, ifos, *args, **kwargs):
        super(AntennaAndTimeManager, self).__init__(*args, **kwargs)
        # If we are sampling over sky position and/or time ...
        self.peak_time_dict = None
        self.antenna_pattern_dict = None
        self.time_delay_dict = None
        if not self.vary_skypos:
            # antenna pattern is fixed if sky location is fixed
            # (it will vary a tiny amount over different geocenter times)
            self.antenna_pattern_dict = rwf.get_antenna_pattern_dict(self.reference_time, ifos,
                                                                     self.fixed['right_ascension'],
                                                                     self.fixed['declination'],
                                                                     self.fixed['polarization'])

            if not self.vary_time:
                # tpeak dict is only fixed if both sky location and time are fixed
                self.peak_time_dict = rwf.get_tgps_dict(
                    self.reference_time, ifos,
                    self.fixed['right_ascension'], self.fixed['declination'])

                self.time_delay_dict = self.compute_time_delay_dict(self.fixed['right_ascension'],
                                                                    self.fixed['declination'],
                                                                    self.reference_time, ifos)
        return

    def get_time_delay_dict(self, x_phys, ifos):
        if self.time_delay_dict is not None:
            return self.time_delay_dict
        return self.compute_time_delay_dict(x_phys['right_ascension'],
                                            x_phys['declination'], x_phys['geocenter_time'], ifos)

    @staticmethod
    def compute_time_delay_dict(right_ascension, declination, geocenter_time, ifos):

        time_delay_dict = {ifo: lal.TimeDelayFromEarthCenter(lal.cached_detector_by_prefix[ifo].location,
                                                             right_ascension,
                                                             declination,
                                                             geocenter_time) for ifo in ifos}
        return time_delay_dict

    def get_tpeak_dict(self, x_phys, ifos):
        if self.peak_time_dict is not None:
            return self.peak_time_dict
        return rwf.get_tgps_dict(x_phys['geocenter_time'], ifos, x_phys['right_ascension'], x_phys['declination'])

    def get_antenna_pattern_dict(self, x_phys, ifos):
        if self.antenna_pattern_dict is not None:
            return self.antenna_pattern_dict
        return rwf.get_antenna_pattern_dict(x_phys['geocenter_time'], ifos,
                                            x_phys['right_ascension'], x_phys['declination'], x_phys['polarization'])


class WaveformManager(LogisticParameterManager):
    def __init__(self, ifos, *args, **kwargs):
        super(WaveformManager, self).__init__(*args, **kwargs)
        self.approx_name = kwargs['approx']
        self.approximant = lalsim.SimInspiralGetApproximantFromString(self.approx_name)
        self.antenna_and_time_manager = AntennaAndTimeManager(ifos, *args, **kwargs)

    def generate_lal_hphc(self, m1_msun, m2_msun, chi1, chi2, delta_t, dist_mpc=1,
                          f22_start=20, f_ref=11, inclination=0, phi_ref=0., eccentricity=0,
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
                                                    delta_t, f22_start, f_ref,
                                                    param_dict,
                                                    self.approximant)
        return hp, hc

    def get_hplus_hcross(self, x_phys, delta_t, f22_start=11, f_ref=11):
        """
        get complex waveform at geocenter
        """
        m1, m2 = m1m2_from_mtotq(x_phys['total_mass'], x_phys['mass_ratio'])
        chi1 = [x_phys['spin1_x'], x_phys['spin1_y'], x_phys['spin1_z']]
        chi2 = [x_phys['spin2_x'], x_phys['spin2_y'], x_phys['spin2_z']]

        hp, hc = self.generate_lal_hphc(m1, m2, chi1, chi2, delta_t=delta_t,
                                        dist_mpc=x_phys['luminosity_distance'],
                                        f22_start=f22_start, f_ref=f_ref,
                                        inclination=x_phys['inclination'],
                                        phi_ref=x_phys['phase'],
                                        eccentricity=x_phys['eccentricity'],
                                        mean_anomaly_periastron=x_phys['mean_anomaly'])
        return TimeSeries.from_lal(hp), TimeSeries.from_lal(hc)

    def get_projected_waveform(self, x_phys, ifos, time_dict, f22_start=11, f_ref=11, window=False):
        delta_t = time_dict[ifos[0]][1] - time_dict[ifos[0]][0]

        # get hplus and hcross
        hp, hc = self.get_hplus_hcross(x_phys, delta_t, f22_start=f22_start, f_ref=f_ref)

        # apply tukey window if desired 
        if window:
            print
            hp = apply_window(hp)
            hc = apply_window(hc)
        
        # set times in geocenter time
        hp.t0 = x_phys['geocenter_time'] + hp.t0.value
        hc.t0 = x_phys['geocenter_time'] + hc.t0.value

        AP_dict = self.antenna_and_time_manager.get_antenna_pattern_dict(x_phys, ifos)
        time_delay_dict = self.antenna_and_time_manager.get_time_delay_dict(x_phys, ifos)

        # Cycle through ifos
        projected_waveform_dict = {}
        for ifo in ifos:
            h_td = hp.value - 1j * hc.value

            Fp, Fc = AP_dict[ifo]

            h_ifo = Fp * h_td.real - Fc * h_td.imag

            # convert h_ifo to detector_time
            time_delay = time_delay_dict[ifo]

            # convert geocenter timeseries to detector timeseries by adding time delay
            # then interpolating that timeseries to the actual, sampled detector times
            h_projected = interpolate_timeseries(hp.times.value + time_delay, h_ifo, time_dict[ifo])
            projected_waveform_dict[ifo] = h_projected

        return projected_waveform_dict


class NewWaveformManager(LogisticParameterManager):
    def __init__(self, ifos, use_higher_order_modes, *args, **kwargs):
        super(NewWaveformManager, self).__init__(*args, **kwargs)
        self.approx_name = kwargs['approx']
        if self.approx_name == 'TEOBResumSDALI':
            if use_higher_order_modes:
                print('generator using higher order modes')
                self.generator = TEOBResumSDALI(modes_to_use=[[2, 2], [2, 1], [3, 3], [4, 4]])
            else:
                self.generator = TEOBResumSDALI(modes_to_use=[[2, 2]])
        else:
            self.generator = gwsignal.core.waveform.LALCompactBinaryCoalescenceGenerator(self.approx_name)

        self.antenna_and_time_manager = AntennaAndTimeManager(ifos, *args, **kwargs)

        return

    def get_hplus_hcross(self, x_phys, delta_t, f22_start=11, f_ref=11):
        """
        get complex waveform at geocenter
        """
        params = self.physical_dict_to_waveform_dict(x_phys)
        params['f22_start'] = f22_start * u.Hz
        params['deltaT'] = delta_t * u.s
        params['f22_ref'] = f_ref * u.Hz
        params['condition'] = 0

        return wfm.GenerateTDWaveform(params, self.generator)

    def get_projected_waveform(self, x_phys, ifos, time_dict, f22_start=11, f_ref=11):
        delta_t = time_dict[ifos[0]][1] - time_dict[ifos[0]][0]

        hp, hc = self.get_hplus_hcross(x_phys, delta_t, f22_start=f22_start, f_ref=f_ref)
        # set times in geocenter time
        hp.t0 = x_phys['geocenter_time'] + hp.t0.value
        hc.t0 = x_phys['geocenter_time'] + hc.t0.value

        AP_dict = self.antenna_and_time_manager.get_antenna_pattern_dict(x_phys, ifos)
        time_delay_dict = self.antenna_and_time_manager.get_time_delay_dict(x_phys, ifos)

        # Cycle through ifos
        projected_waveform_dict = {}
        for ifo in ifos:
            h_td = hp.value - 1j * hc.value

            Fp, Fc = AP_dict[ifo]

            h_ifo = Fp * h_td.real - Fc * h_td.imag

            # convert h_ifo to detector_time
            time_delay = time_delay_dict[ifo]

            # convert geocenter timeseries to detector timeseries by adding time delay
            # then interpolating that timeseries to the actual, sampled detector times
            h_projected = interpolate_timeseries(hp.times.value + time_delay, h_ifo, time_dict[ifo])
            projected_waveform_dict[ifo] = h_projected

        return projected_waveform_dict


class LnLikelihoodManager(LogisticParameterManager):
    def __init__(self, psd_dict, time_dict, data_dict, f_low, f_ref, f22_start, f_max=None, only_prior=False,
                 use_higher_order_modes=False, *args, **kwargs):
        self.time_dict = time_dict
        self.data_dict = data_dict
        self.f_low = f_low
        self.f_ref = f_ref
        self.f22_start = f22_start
        self.f_max = f_max
        self.psd_dict = psd_dict
        self.rho_dict = self._make_autocorrolation_dict()
        self.ifos = list(self.data_dict.keys())
        for ifo, rho in self.rho_dict.items():
            assert len(rho) == len(self.data_dict[ifo]), 'Length for ACF is not the same as for the data'
        self.only_prior = only_prior
        try:
            self.waveform_manager = NewWaveformManager(self.ifos, use_higher_order_modes, *args, **kwargs)
        except Exception as e:
            print(e)
            print("warning, new waveform manager has failed to be created, using old waveform manager")
            self.waveform_manager = WaveformManager(self.ifos, *args, **kwargs)
        self.log_prior = LnPriorManager(*args, **kwargs)

        super().__init__(*args, **kwargs)

    def _make_autocorrolation_dict(self):
        return get_ACF(self.psd_dict, self.time_dict, f_low=self.f_low, f_max=self.f_max)

    def waveform_has_error(self, phys_dict, waveform_ifo, residual):
        # check for various errors before continuing
        if np.all(waveform_ifo == 0):
            print('waveform falls outside allowed window for:')
            print(phys_dict)
            return False

        if sum(np.isnan(residual)) > 0:
            print('NaNs in residuals for:')
            print(phys_dict)
            return True

        if sum(np.isinf(residual)) > 0:
            print('infinities in residuals for:')
            print(phys_dict)
            return True

        return False

    def get_log_posterior(self, x_phys, verbose=False, **kwargs):
        if verbose:
            print('getting wf')
        
        projected_wf_dict = self.waveform_manager.get_projected_waveform(
            x_phys, self.ifos, self.time_dict,
            f22_start=kwargs.get('f22_start', self.f22_start),
            f_ref=kwargs.get('f_ref', self.f_ref),
            window=kwargs.get('window', False)
        )
        if verbose:
            print('done getting wf')

        ln_posterior = 0

        # Cycle through ifos
        for ifo, data in self.data_dict.items():

            # Truncate and compute residuals
            r = data - projected_wf_dict[ifo]

            if self.waveform_has_error(x_phys, projected_wf_dict[ifo], r):
                return -np.inf

            # "Over whiten" residuals
            rwt = solve_toeplitz(self.rho_dict[ifo], r)

            # Compute log likelihood for ifo
            ln_posterior -= 0.5 * np.dot(r, rwt)
        return ln_posterior

    def get_lnprob(self, x, verbose=False,
                   **kwargs):
        # get physical parameters
        x_phys = kwargs.pop('x_phys', None)
        if x_phys is None:
            x_phys = self.samp_to_phys(x)
        if verbose:
            print('x_phys', x_phys)

        # Initialize posterior to 0
        lnprob = 0

        # Calculate posterior
        if not self.only_prior:
            lnprob += self.get_log_posterior(x_phys, verbose=verbose)

        # Calculate prior
        lnprob += self.log_prior.get_lnprior(x, phys_dict=x_phys)

        # Check for NaN
        if lnprob != lnprob:
            print('lnprob = NaN for:')
            print(x_phys)
            return -np.inf

        # Return posterior
        return lnprob
            
    def get_SNRs(self, samples): 
        
        '''
        Get SNRs for a list of samples
        '''
                
        # set up arrays
        per_detector_opt_snrs = np.zeros((len(self.ifos), len(samples)))
        per_detector_mf_snrs = np.zeros((len(self.ifos), len(samples)))
        network_opt_snrs = np.zeros(len(samples))
        network_mf_snrs = np.zeros(len(samples))
        
        # cycle through the input samples
        samples_phys = [self.samp_to_phys(x) for x in samples]
        for i,x_phys in enumerate(samples_phys):
            
            # get waveforms for this sample
            projected_wf_dict = self.waveform_manager.get_projected_waveform(
                    x_phys, self.ifos, self.time_dict,
                    f22_start=self.f22_start,
                    f_ref=self.f_ref
                )
            
            # lists for per-detector snrs
            opt_snrs = []
            mf_snrs = []
            
            # cycle through the interferometers
            for ifo in self.ifos:
            
                sig = projected_wf_dict[ifo]
                data = self.data_dict[ifo]
                rho = self.rho_dict[ifo]
                
                opt_snrs.append(calc_opt_SNR(sig, rho))
                mf_snrs.append(calc_mf_SNR(data, sig, rho))
                
            # add to arrays 
            per_detector_opt_snrs[:,i] = opt_snrs
            per_detector_mf_snrs[:,i] = mf_snrs
            
            # calculate network snrs
            network_opt_snrs[i] = calc_network_SNR(opt_snrs)
            network_mf_snrs[i] = calc_network_SNR(mf_snrs)
        
        # create dict with all and return 
        SNRs_dict = {
            'network_optimal_SNR':network_opt_snrs,
            'network_matched_filter_SNR':network_mf_snrs,
            **{f'{ifo}_optimal_SNR':per_detector_opt_snrs[i] for i,ifo in enumerate(self.ifos)}, 
            **{f'{ifo}_matched_filter_SNR':per_detector_mf_snrs[i] for i,ifo in enumerate(self.ifos)}
        }
        return SNRs_dict
