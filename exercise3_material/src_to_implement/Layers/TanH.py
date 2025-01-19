import numpy as np
from .Base import BaseLayer  # Updated this line

class TanH(BaseLayer):
    def __init__(self):
        """
        TanH activation function.
        """
        super().__init__()
        self.activations = None

    def forward(self, input_tensor):
        """
        Forward pass for TanH.
        :param input_tensor: Input tensor.
        :return: Output tensor after applying TanH.
        """
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):
        """
        Backward pass for TanH.
        :param error_tensor: Gradient of the loss with respect to the output.
        :return: Gradient of the loss with respect to the input.
        """
        return error_tensor * (1 - self.activations ** 2)