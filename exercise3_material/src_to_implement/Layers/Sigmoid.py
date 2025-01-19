import numpy as np
from .Base import BaseLayer  # Updated this line

class Sigmoid(BaseLayer):
    def __init__(self):
        """
        Sigmoid activation function.
        """
        super().__init__()
        self.activations = None

    def forward(self, input_tensor):
        """
        Forward pass for Sigmoid.
        :param input_tensor: Input tensor.
        :return: Output tensor after applying Sigmoid.
        """
        self.activations = 1 / (1 + np.exp(-input_tensor))
        return self.activations

    def backward(self, error_tensor):
        """
        Backward pass for Sigmoid.
        :param error_tensor: Gradient of the loss with respect to the output.
        :return: Gradient of the loss with respect to the input.
        """
        return error_tensor * self.activations * (1 - self.activations)