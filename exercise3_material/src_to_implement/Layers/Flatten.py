import numpy as np
from .Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        """
        Flatten Layer.
        """
        super().__init__()
        self.input_shape = None

    def forward(self, input_tensor):
        """
        Forward pass for the Flatten layer.
        :param input_tensor: Input tensor.
        :return: Flattened output tensor.
        """
        self.input_shape = input_tensor.shape  # Store input shape for backward pass
        
        # Reshape the input tensor
        batch_size = input_tensor.shape[0]
        output_tensor = input_tensor.reshape(batch_size, -1)  # Flatten all dimensions except batch

        # Debug: Print the shape of the output tensor
        print("Output tensor shape in Flatten:", output_tensor.shape)

        return output_tensor

        #return input_tensor.reshape(input_tensor.shape[0], -1)  # Flatten to (batch_size, -1)

    def backward(self, error_tensor):
        """
        Backward pass for the Flatten layer.
        :param error_tensor: Gradient of the loss with respect to the output.
        :return: Gradient of the loss with respect to the input.
        """
        return error_tensor.reshape(self.input_shape)  # Reshape to original input shape