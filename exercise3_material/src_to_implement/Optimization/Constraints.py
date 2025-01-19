import numpy as np

class L1_Regularizer:
    def __init__(self, alpha):
        """
        L1 Regularizer.
        :param alpha: Regularization strength.
        """
        self.alpha = alpha

    def calculate_gradient(self, weights):
        """
        Calculates the gradient of the L1 regularization term.
        :param weights: Weights of the layer.
        :return: Gradient of the L1 regularization term.
        """
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        """
        Calculates the L1 norm of the weights.
        :param weights: Weights of the layer.
        :return: L1 norm of the weights.
        """
        return self.alpha * np.sum(np.abs(weights))

class L2_Regularizer:
    def __init__(self, alpha):
        """
        L2 Regularizer.
        :param alpha: Regularization strength.
        """
        self.alpha = alpha

    def calculate_gradient(self, weights):
        """
        Calculates the gradient of the L2 regularization term.
        :param weights: Weights of the layer.
        :return: Gradient of the L2 regularization term.
        """
        return self.alpha * weights

    def norm(self, weights):
        """
        Calculates the L2 norm of the weights.
        :param weights: Weights of the layer.
        :return: L2 norm of the weights.
        """
        return self.alpha * np.sum(weights ** 2)