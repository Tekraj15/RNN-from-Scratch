import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        """
        CrossEntropyLoss function.
        """
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        """
        Forward pass for CrossEntropyLoss.
        :param input_tensor: Predicted probabilities (output of SoftMax).
        :param label_tensor: Ground truth labels (one-hot encoded).
        :return: Cross-entropy loss.
        """
        self.input_tensor = input_tensor
        epsilon = 1e-15  # Avoid log(0)
        clipped_input = np.clip(input_tensor, epsilon, 1 - epsilon)
        loss = -np.sum(label_tensor * np.log(clipped_input)) / input_tensor.shape[0]
        return loss

    def backward(self, label_tensor):
        """
        Backward pass for CrossEntropyLoss.
        :param label_tensor: Ground truth labels (one-hot encoded).
        :return: Gradient of the loss with respect to the input.
        """
        return (self.input_tensor - label_tensor) / self.input_tensor.shape[0]