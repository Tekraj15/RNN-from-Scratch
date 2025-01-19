import numpy as np
from .Base import BaseLayer

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        """
        Batch Normalization layer.
        :param channels: Number of channels in the input tensor.
        """
        super().__init__()
        self.channels = channels
        self.trainable = True
        self.weights = np.ones(channels)  # Gamma (scale)
        self.bias = np.zeros(channels)  # Beta (shift)
        self.epsilon = 1e-10
        self.mean = None
        self.variance = None
        self.moving_mean = np.zeros(channels)
        self.moving_variance = np.zeros(channels)
        self.momentum = 0.9
        self.input_tensor = None  # Store input tensor for backward pass
        self.normalized_input = None  # Store normalized input for backward pass
        self.testing_phase = False  # Default to False (training mode)

    def reformat(self, tensor):
        """
        Reformat the tensor between 2D (vector) and 4D (image) shapes.
        :param tensor: Input tensor.
        :return: Reformatted tensor.
        """
        if len(tensor.shape) == 4:
            # Convert 4D (batch, channels, height, width) to 2D (batch * height * width, channels)
            batch_size, channels, height, width = tensor.shape
            return tensor.transpose(0, 2, 3, 1).reshape(-1, channels)
        elif len(tensor.shape) == 2:
            # Convert 2D (batch * height * width, channels) to 4D (batch, channels, height, width)
            batch_size_times_spatial = tensor.shape[0]
            channels = tensor.shape[1]
            batch_size = self.input_tensor.shape[0]  # Use original input shape
            height = self.input_tensor.shape[2]
            width = self.input_tensor.shape[3]
            return tensor.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)
        else:
            raise ValueError("Input tensor must be 2D or 4D.")

    def forward(self, input_tensor):
        """
        Forward pass for Batch Normalization.
        :param input_tensor: Input tensor.
        :return: Normalized output tensor.
        """
        self.input_tensor = input_tensor  # Store input tensor for backward pass

        if len(input_tensor.shape) == 4:
            # For convolutional layers, reformat to 2D
            input_tensor = self.reformat(input_tensor)

        if not self.testing_phase:
            # Compute mean and variance for the current batch
            self.mean = np.mean(input_tensor, axis=0)
            self.variance = np.var(input_tensor, axis=0)

            # Update moving mean and variance
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.mean
            self.moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * self.variance

            # Normalize the input
            self.normalized_input = (input_tensor - self.mean) / np.sqrt(self.variance + self.epsilon)
        else:
            # Use moving mean and variance during testing
            self.normalized_input = (input_tensor - self.moving_mean) / np.sqrt(self.moving_variance + self.epsilon)

        # Scale and shift
        output_tensor = self.weights * self.normalized_input + self.bias

        if len(self.input_tensor.shape) == 4:
            # For convolutional layers, reformat back to 4D
            output_tensor = self.reformat(output_tensor)

        return output_tensor

    def backward(self, error_tensor):
        """
        Backward pass for Batch Normalization.
        :param error_tensor: Gradient of the loss with respect to the output.
        :return: Gradient of the loss with respect to the input.
        """
        if len(error_tensor.shape) == 4:
            # For convolutional layers, reformat to 2D
            error_tensor = self.reformat(error_tensor)

        # Compute gradients for weights and bias
        self.gradient_weights = np.sum(error_tensor * self.normalized_input, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)

        # Compute gradient with respect to the input
        N = error_tensor.shape[0]
        sigma = np.sqrt(self.variance + self.epsilon)
        grad_normalized = error_tensor * self.weights
        grad_variance = np.sum(grad_normalized * (self.input_tensor - self.mean.reshape(1, -1)) * -0.5 * (self.variance + self.epsilon) ** (-1.5), axis=0)
        grad_mean = np.sum(grad_normalized * -1 / sigma, axis=0) + grad_variance * np.sum(-2 * (self.input_tensor - self.mean), axis=0) / N
        grad_input = (grad_normalized / sigma) + (grad_variance * 2 * (self.input_tensor - self.mean) / N) + (grad_mean / N)

        if len(self.input_tensor.shape) == 4:
            # For convolutional layers, reformat back to 4D
            grad_input = self.reformat(grad_input)

        return grad_input
