import numpy as np
from .Base import BaseLayer  # Updated this line

class Dropout(BaseLayer):
    def __init__(self, probability):
        """
        Dropout layer.
        :param probability: Probability of keeping a unit active (1 - dropout rate).
        """
        super().__init__()
        self.probability = probability
        self.mask = None

    def forward(self, input_tensor):
        """
        Forward pass for Dropout.
        :param input_tensor: Input tensor.
        :return: Output tensor after applying dropout.
        """
        if not self.testing_phase:
            # Generate a mask to drop units
            self.mask = (np.random.rand(*input_tensor.shape) < self.probability) / self.probability
            return input_tensor * self.mask
        else:
            # During testing, no dropout is applied
            return input_tensor

    def backward(self, error_tensor):
        """
        Backward pass for Dropout.
        :param error_tensor: Gradient of the loss with respect to the output.
        :return: Gradient of the loss with respect to the input.
        """
        if not self.testing_phase:
            return error_tensor * self.mask
        else:
            return error_tensor