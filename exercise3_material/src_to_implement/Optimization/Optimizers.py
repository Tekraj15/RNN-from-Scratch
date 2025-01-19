import numpy as np
from Optimization.BaseOptimizer import BaseOptimizer

class Sgd(BaseOptimizer):
    def __init__(self, learning_rate):
        """
        Stochastic Gradient Descent (SGD) optimizer.
        :param learning_rate: Learning rate for the optimizer.
        """
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Calculates the updated weights using SGD.
        :param weight_tensor: Current weights.
        :param gradient_tensor: Gradient of the loss with respect to the weights.
        :return: Updated weights.
        """
        if self.regularizer:
            # Apply the regularizer gradient to the gradient tensor
            gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)
        
        # Update the weights using the learning rate
        return weight_tensor - self.learning_rate * gradient_tensor

class SgdWithMomentum(BaseOptimizer):
    def __init__(self, learning_rate, momentum_rate):
        """
        SGD with Momentum optimizer.
        :param learning_rate: Learning rate for the optimizer.
        :param momentum_rate: Momentum rate.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None  # Velocity

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Calculates the updated weights using SGD with Momentum.
        :param weight_tensor: Current weights.
        :param gradient_tensor: Gradient of the loss with respect to the weights.
        :return: Updated weights.
        """
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)

        if self.regularizer:
            # Apply the regularizer gradient to the gradient tensor
            gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)

        # Update the velocity and weights using momentum
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        return weight_tensor + self.v

class Adam(BaseOptimizer):
    def __init__(self, learning_rate, mu, rho):
        """
        Adam optimizer.
        :param learning_rate: Learning rate for the optimizer.
        :param mu: Decay rate for the first moment (mean).
        :param rho: Decay rate for the second moment (variance).
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = None  # First moment (mean)
        self.r = None  # Second moment (variance)
        self.t = 0  # Time step

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Calculates the updated weights using Adam.
        :param weight_tensor: Current weights.
        :param gradient_tensor: Gradient of the loss with respect to the weights.
        :return: Updated weights.
        """
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
        if self.r is None:
            self.r = np.zeros_like(weight_tensor)

        if self.regularizer:
            # Apply the regularizer gradient to the gradient tensor
            gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)

        self.t += 1
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * gradient_tensor ** 2

        # Correct the bias terms
        v_hat = self.v / (1 - self.mu ** self.t)
        r_hat = self.r / (1 - self.rho ** self.t)

        # Update the weights
        return weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + 1e-8)
