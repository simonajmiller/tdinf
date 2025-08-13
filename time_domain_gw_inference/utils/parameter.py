import numpy as np
from .misc import inv_logit, logit, logit_jacobian


class LogisticParameter:
    """
    Represents a model parameter transformed via a logistic mapping.

    The logistic transform maps values from a finite physical range
    (xmin, xmax) to the entire real line (-∞, ∞), which can improve
    numerical behavior in optimization or sampling.

    Parameters
    ----------
    physical_name : str
        Name of the parameter in physical space.
    limit : tuple of float
        (xmin, xmax) limits of the parameter in physical space.
    initial_value : float, optional
        Initial guess for the parameter in physical space. If None,
        the initial logistic value is drawn from a standard normal.
    """
    def __init__(self, physical_name, limit, initial_value=None):
        self.physical_name = physical_name
        self.logistic_name = 'x_' + physical_name
        self.limit = limit
        self.initial_value = initial_value

    def logistic_to_physical(self, value):
        """
        Convert a logistic-space value to physical space.
        """
        return inv_logit(value, *self.limit)

    def physical_to_logistic(self, physical_value):
        """
        Convert a physical-space value to logistic space.
        """
        return logit(physical_value, *self.limit)

    def ln_prior_weight(self, physical_value):
        """
        Get required prior weight for uniform, flat prior in non-logistic space
        """
        return -np.log(logit_jacobian(physical_value, *self.limit))

    def initialize_value(self, scale=0.05):
        """
        Initialize the parameter in logistic space.

        If `initial_value` is set, draw from a normal distribution centered
        at its logistic transform; otherwise draw from a standard normal.

        Parameters
        ----------
        scale : float, optional
            Standard deviation for the Gaussian initialization. Default is 0.05.

        Returns
        -------
        float
            Initial value in logistic space.
        """
        if self.initial_value is not None:
            logistic_initial_value = self.physical_to_logistic(self.initial_value)
            return np.random.normal(loc=logistic_initial_value, scale=scale)
        return np.random.normal()


class TrigLogisticParameter:
    """
    Represents a model parameter where a trigonometric function of the parameter
    is transformed via a logistic mapping. Useful for angular parameters.

    Parameters
    ----------
    physical_name : str
        Name of the parameter in physical space.
    trig_function : {'cos', 'sin'}
        Which trigonometric function to apply before the logistic transform.
    limit : tuple of float
        (xmin, xmax) limits in the transformed space.
    initial_value : float, optional
        Initial guess for the parameter in physical space.
    """
    def __init__(self, physical_name, trig_function, limit, initial_value=None):
        if trig_function == 'cos':
            self.trig_function = np.cos
            self.inverse_trig = np.arccos
        elif trig_function == 'sin':
            self.trig_function = np.sin
            self.inverse_trig = np.arcsin
        else:
            raise ValueError(f'TrigLogisticParameter must be given "cos" or "sin" as trig function arguments, given {trig_function}')
        self.physical_name = physical_name
        self.trig_name = trig_function + '_' + physical_name
        self.logistic_name = 'x_' + self.trig_name
        self.limit = limit
        self.initial_value = initial_value

    def logistic_to_physical(self, value):
        """
        Convert a logistic-space value to an angle in radians.
        """
        return self.inverse_trig(inv_logit(value, *self.limit))

    def physical_to_logistic(self, physical_value):
        """
        Convert an angle in radians to logistic space by applying the
        trigonometric function first.
        """
        return logit(self.trig_function(physical_value), *self.limit)

    def ln_prior_weight(self, physical_value):
        """
        Compute the log of the prior weight for a flat prior in physical space,
        accounting for the trigonometric mapping and logistic Jacobian.
        """
        return -np.log(logit_jacobian(self.trig_function(physical_value), *self.limit))


class CartesianAngle:
    """
    Represents an angle parameterized in Cartesian form (x, y).

    This class provides conversions between Cartesian coordinates
    (x, y) and an angle in radians, along with a Gaussian prior in Cartesian space.

    Parameters
    ----------
    physical_name : str
        Base name of the parameter.
    phase_offset : float, optional
        Offset to add to angles in radians when converting from Cartesian form.
        Default is 0.
    """
    def __init__(self, physical_name, phase_offset=0):
        self.physical_name = physical_name
        self.x_name = physical_name + '_x'
        self.y_name = physical_name + '_y'
        self.phase_offset = phase_offset

    def cartesian_to_radian(self, value_x, value_y):
        """
        Convert Cartesian coordinates (x, y) to an angle in radians,
        applying the phase offset.
        """
        return np.arctan2(value_y, value_x) + self.phase_offset

    def radian_to_cartesian(self, value):
        """
        Convert an angle in radians to Cartesian coordinates (cos θ, sin θ).
        """
        return np.cos(value), np.sin(value)

    def ln_prior_weight(self, value_x, value_y):
         """
        Compute the log of prior weight in Cartesian space.
        """
        return -0.5 * (value_x ** 2 + value_y ** 2)

    def initialize_value(self):
        """
        Initialize (x, y) values from independent standard normal distributions.

        Returns
        -------
        tuple of float
            Randomly drawn (x, y) coordinates.
        """
        return np.random.normal(), np.random.normal()








