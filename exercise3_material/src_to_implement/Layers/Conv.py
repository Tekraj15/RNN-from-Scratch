import numpy as np
from .Base import BaseLayer

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        """
        Convolutional Layer.
        :param stride_shape: Stride shape (height, width).
        :param convolution_shape: Shape of the convolution kernel (input_channels, kernel_height, kernel_width).
        :param num_kernels: Number of output channels (kernels).
        """
        super().__init__()
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True

        # Initialize weights and biases
        self.weights = np.random.randn(num_kernels, *convolution_shape) * 0.01
        self.bias = np.zeros(num_kernels)

        # Debug: Print the shape of bias
        print("Bias shape after initialization:", self.bias.shape)
        print("Bias value after initialization:", self.bias)

        # Gradients
        self.gradient_weights = None
        self.gradient_bias = None
        self.input_tensor = None

    def forward(self, input_tensor):
        """
        Forward pass for the Conv layer.
        :param input_tensor: Input tensor.
        :return: Output tensor.
        """
        self.input_tensor = input_tensor  # Store input for backward pass
        batch_size, input_channels, input_height, input_width = input_tensor.shape
        output_height = (input_height - self.convolution_shape[1]) // self.stride_shape[0] + 1
        output_width = (input_width - self.convolution_shape[2]) // self.stride_shape[1] + 1

        # Initialize output tensor
        output_tensor = np.zeros((batch_size, self.num_kernels, output_height, output_width))

        # Perform convolution
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * self.stride_shape[0]
                        h_end = h_start + self.convolution_shape[1]
                        w_start = j * self.stride_shape[1]
                        w_end = w_start + self.convolution_shape[2]

                        # Extract the input patch
                        input_patch = input_tensor[b, :, h_start:h_end, w_start:w_end]



                        # Compute the convolution
                        #output_tensor[b, k, i, j] = np.sum(input_patch * self.weights[k]) + self.bias[k]

                        convolution_result = np.sum(input_patch * self.weights[k])
                        bias_term = self.bias[k]  # Ensure this is a scalar
                        bias_term = float(self.bias[k])  # Explicitly convert to scalar
                        output_tensor[b, k, i, j] = convolution_result + bias_term

        return output_tensor

    def backward(self, error_tensor):
        """
        Backward pass for the Conv layer.
        :param error_tensor: Gradient of the loss with respect to the output.
        :return: Gradient of the loss with respect to the input.
        """
        batch_size, input_channels, input_height, input_width = self.input_tensor.shape
        output_height, output_width = error_tensor.shape[2], error_tensor.shape[3]

        # Initialize gradients
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)
        gradient_input = np.zeros_like(self.input_tensor)

        # Compute gradients
        for b in range(batch_size):
            for k in range(self.num_kernels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * self.stride_shape[0]
                        h_end = h_start + self.convolution_shape[1]
                        w_start = j * self.stride_shape[1]
                        w_end = w_start + self.convolution_shape[2]

                        # Extract the input patch
                        input_patch = self.input_tensor[b, :, h_start:h_end, w_start:w_end]

                        # Compute gradients for weights and biases
                        self.gradient_weights[k] += input_patch * error_tensor[b, k, i, j]
                        self.gradient_bias[k] += error_tensor[b, k, i, j]

                        # Compute gradient with respect to the input
                        gradient_input[b, :, h_start:h_end, w_start:w_end] += self.weights[k] * error_tensor[b, k, i, j]

        return gradient_input

    def initialize(self, weights_initializer, bias_initializer):
        """
        Initialize weights and biases.
        :param weights_initializer: Initializer for weights.
        :param bias_initializer: Initializer for biases.
        """
        self.weights = weights_initializer.initialize(
            (self.num_kernels, *self.convolution_shape), self.convolution_shape[0], self.num_kernels
        )
        self.bias = bias_initializer.initialize(
            (self.num_kernels), self.convolution_shape[0], self.num_kernels
        )
