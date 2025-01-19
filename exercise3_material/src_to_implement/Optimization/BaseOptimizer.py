class BaseOptimizer:
    def __init__(self):
        """
        Base class for all optimizers.
        """
        self.regularizer = None  # Regularizer (e.g., L1, L2)

    def add_regularizer(self, regularizer):
        """
        Adds a regularizer to the optimizer.
        :param regularizer: Regularizer object (e.g., L1_Regularizer, L2_Regularizer).
        """
        self.regularizer = regularizer

    def calculate_update(self, weight_tensor, gradient_tensor):
        """
        Calculates the updated weights.
        :param weight_tensor: Current weights.
        :param gradient_tensor: Gradient of the loss with respect to the weights.
        :return: Updated weights.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")
