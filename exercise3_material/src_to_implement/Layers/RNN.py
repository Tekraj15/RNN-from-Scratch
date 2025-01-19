import numpy as np
from .Base import BaseLayer  # Updated this line

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        """
        RNN layer.
        :param input_size: Size of the input vector.
        :param hidden_size: Size of the hidden state.
        :param output_size: Size of the output vector.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)
        self.memorize = False  # Whether to carry over hidden state between sequences
        self.trainable = True

        # Initialize weights and biases
        self.weights_input = np.random.randn(input_size, hidden_size)
        self.weights_hidden = np.random.randn(hidden_size, hidden_size)
        self.weights_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

    def forward(self, input_tensor):
        """
        Forward pass for RNN.
        :param input_tensor: Input tensor.
        :return: Output tensor.
        """
        batch_size = input_tensor.shape[0]
        outputs = np.zeros((batch_size, self.output_size))

        for t in range(batch_size):
            # Update hidden state
            self.hidden_state = np.tanh(
                np.dot(input_tensor[t], self.weights_input) +
                np.dot(self.hidden_state, self.weights_hidden) +
                self.bias_hidden
            )
            # Compute output
            outputs[t] = np.dot(self.hidden_state, self.weights_output) + self.bias_output

        return outputs

    def backward(self, error_tensor):
        """
        Backward pass for RNN.
        :param error_tensor: Gradient of the loss with respect to the output.
        :return: Gradient of the loss with respect to the input.
        """
        # Not implemented for brevity
        raise NotImplementedError("Backward pass for RNN is not implemented.")