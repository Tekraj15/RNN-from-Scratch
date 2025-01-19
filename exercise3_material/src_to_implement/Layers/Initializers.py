import numpy as np

class UniformRandom:
    def __init__(self, low=-0.1, high=0.1):
        """
        Uniform Random Initializer.
        :param low: Lower bound of the uniform distribution.
        :param high: Upper bound of the uniform distribution.
        """
        self.low = low
        self.high = high

    def initialize(self, shape, fan_in, fan_out):
        """
        Initialize weights using a uniform distribution.
        :param shape: Shape of the weights.
        :param fan_in: Number of input units.
        :param fan_out: Number of output units.
        :return: Initialized weights.
        """
        return np.random.uniform(self.low, self.high, shape)

class Xavier:
    def __init__(self):
        """
        Xavier Initializer.
        """
        pass

    def initialize(self, shape, fan_in, fan_out):
        """
        Initialize weights using Xavier initialization.
        :param shape: Shape of the weights.
        :param fan_in: Number of input units.
        :param fan_out: Number of output units.
        :return: Initialized weights.
        """
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0, scale, shape)

class He:
    def __init__(self):
        """
        He Initializer.
        """
        pass

    def initialize(self, shape, fan_in, fan_out):
        """
        Initialize weights using He initialization.
        :param shape: Shape of the weights.
        :param fan_in: Number of input units.
        :param fan_out: Number of output units.
        :return: Initialized weights.
        """
        scale = np.sqrt(2.0 / fan_in)
        return np.random.normal(0, scale, shape)

class Constant:
    def __init__(self, value=0.1):
        """
        Constant Initializer.
        :param value: Constant value to initialize weights or biases.
        """
        self.value = value

    def initialize(self, shape, fan_in, fan_out):
        """
        Initialize weights or biases with a constant value.
        :param shape: Shape of the weights or biases.
        :param fan_in: Number of input units.
        :param fan_out: Number of output units.
        :return: Initialized weights or biases.
        """
        return np.full(shape, self.value)