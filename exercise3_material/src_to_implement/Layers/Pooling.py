import numpy as np
from .Base import BaseLayer

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        """
        Pooling Layer.
        :param stride_shape: Stride shape (height, width).
        :param pooling_shape: Shape of the pooling window (height, width).
        """
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None

    def forward(self, input_tensor):
        """
        Forward pass for the Pooling layer.
        :param input_tensor: Input tensor.
        :return: Output tensor.
        """
        self.input_tensor = input_tensor  # Store input for backward pass
        batch_size, input_channels, input_height, input_width = input_tensor.shape
        output_height = (input_height - self.pooling_shape[0]) // self.stride_shape[0] + 1
        output_width = (input_width - self.pooling_shape[1]) // self.stride_shape[1] + 1

        # Initialize output tensor
        output_tensor = np.zeros((batch_size, input_channels, output_height, output_width))

        # Perform max pooling
        for b in range(batch_size):
            for c in range(input_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * self.stride_shape[0]
                        h_end = h_start + self.pooling_shape[0]
                        w_start = j * self.stride_shape[1]
                        w_end = w_start + self.pooling_shape[1]

                        # Extract the input patch
                        input_patch = input_tensor[b, c, h_start:h_end, w_start:w_end]

                        # Compute the max value
                        output_tensor[b, c, i, j] = np.max(input_patch)

        return output_tensor

    def backward(self, error_tensor):
        """
        Backward pass for the Pooling layer.
        :param error_tensor: Gradient of the loss with respect to the output.
        :return: Gradient of the loss with respect to the input.
        """
        batch_size, input_channels, input_height, input_width = self.input_tensor.shape
        output_height, output_width = error_tensor.shape[2], error_tensor.shape[3]

        # Initialize gradient with respect to the input
        gradient_input = np.zeros_like(self.input_tensor)

        # Compute gradients
        for b in range(batch_size):
            for c in range(input_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * self.stride_shape[0]
                        h_end = h_start + self.pooling_shape[0]
                        w_start = j * self.stride_shape[1]
                        w_end = w_start + self.pooling_shape[1]

                        # Extract the input patch
                        input_patch = self.input_tensor[b, c, h_start:h_end, w_start:w_end]

                        # Find the position of the max value
                        max_pos = np.unravel_index(np.argmax(input_patch), input_patch.shape)

                        # Propagate the gradient to the max value position
                        gradient_input[b, c, h_start + max_pos[0], w_start + max_pos[1]] += error_tensor[b, c, i, j]

        return gradient_input