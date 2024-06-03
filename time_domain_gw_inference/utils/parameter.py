import numpy as np
from .misc import inv_logit, logit, logit_jacobian


class LogisticParameter:
    """
    Logistic transformations take values to +/- infinity
    """
    def __init__(self, physical_name, limit, initial_value=None):
        self.physical_name = physical_name
        self.logistic_name = 'x_' + physical_name
        self.limit = limit
        self.initial_value = initial_value

    def logistic_to_physical(self, value):
        return inv_logit(value, *self.limit)

    def physical_to_logistic(self, physical_value):
        return logit(physical_value, *self.limit)

    def ln_prior_weight(self, physical_value):
        """
        Get required prior weight for uniform, flat prior in non-logistic space
        :param physical_value:
        :return:
        """
        return -np.log(logit_jacobian(physical_value, *self.limit))

    def initialize_value(self, scale=0.05):
        if self.initial_value is not None:
            logistic_initial_value = self.physical_to_logistic(self.initial_value)
            return np.random.normal(loc=logistic_initial_value, scale=scale)
        return np.random.normal()


class TrigLogisticParameter:
    """
    Logistic transformations take values to +/- infinity (?)
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
        Converts to angle
        """
        return self.inverse_trig(inv_logit(value, *self.limit))

    def physical_to_logistic(self, physical_value):
        return logit(self.trig_function(physical_value), *self.limit)

    def ln_prior_weight(self, physical_value):
        return -np.log(logit_jacobian(self.trig_function(physical_value),
                                         *self.limit))


class CartesianAngle:
    """
    Often it is easier to sample in the cartesian angles than to sample in one angle? so we do this?
    """
    def __init__(self, physical_name, phase_offset=0):
        self.physical_name = physical_name
        self.x_name = physical_name + '_x'
        self.y_name = physical_name + '_y'
        self.phase_offset = phase_offset

    def cartesian_to_radian(self, value_x, value_y):
        return np.arctan2(value_y, value_x) + self.phase_offset

    def radian_to_cartesian(self, value):
        # Note, the amplitude on this will just be set to a unit vector since this
        # is honestly not a real inverse function
        return np.cos(value), np.sin(value)

    def ln_prior_weight(self, value_x, value_y):
        return -0.5 * (value_x ** 2 + value_y ** 2)

    def initialize_value(self):
        return np.random.normal(), np.random.normal()








