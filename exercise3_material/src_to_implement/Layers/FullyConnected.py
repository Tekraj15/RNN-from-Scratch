import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        """
        Fully Connected (Dense) Layer.
        :param input_size: Number of input features.
        :param output_size: Number of output features.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True

        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros(output_size)

        # Gradients
        self.gradient_weights = None
        self.gradient_bias = None
        self.input_tensor = None  # Store input for backward pass

    def forward(self, input_tensor):
        """
        Forward pass for the Fully Connected layer.
        :param input_tensor: Input tensor.
        :return: Output tensor.
        """
        self.input_tensor = input_tensor  # Store input for backward pass



        # # Changes for Testing the shape
                         
        # print("input_patch shape:", input_patch.shape)
        # print("weights[k] shape:", self.weights[k].shape)

        # # Testing 2
        # # Ensure input_patch and weights[k] have the same shape
        # if input_patch.shape != self.weights[k].shape:
        #     raise ValueError(f"Shape mismatch: input_patch {input_patch.shape}, weights[k] {self.weights[k].shape}")

        # # Debug: Print the shape and type of bias[k]
        # print(f"Bias[{k}] shape: {self.bias[k].shape}, type: {type(self.bias[k])}")

        # # To check the value of k and the shape of self.bias[k]
        # print(f"Index k: {k}, Bias[k] shape: {self.bias[k].shape}, Bias[k] value: {self.bias[k]}")

        
        # Debug: Print the shape of input tensor
        print("Input tensor shape in FullyConnected:", input_tensor.shape)

        return np.dot(input_tensor, self.weights) + self.bias

    def backward(self, error_tensor):
        """
        Backward pass for the Fully Connected layer.
        :param error_tensor: Gradient of the loss with respect to the output.
        :return: Gradient of the loss with respect to the input.
        """
        # Compute gradients for weights and biases
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        self.gradient_bias = np.sum(error_tensor, axis=0)

        # Compute gradient with respect to the input
        return np.dot(error_tensor, self.weights.T)

    def initialize(self, weights_initializer, bias_initializer):
        """
        Initialize weights and biases.
        :param weights_initializer: Initializer for weights.
        :param bias_initializer: Initializer for biases.
        """
        self.weights = weights_initializer.initialize(
            (self.input_size, self.output_size), self.input_size, self.output_size
        )
        self.bias = bias_initializer.initialize(
            (1, self.output_size), self.input_size, self.output_size
        )
        
        # Debug: Print the shape of weights
        print("Weights shape in FullyConnected:", self.weights.shape)