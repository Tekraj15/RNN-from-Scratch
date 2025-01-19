import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        exp_input = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        self.output_tensor = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        return self.output_tensor

    def backward(self, error_tensor):
        return error_tensor * (self.output_tensor - np.square(self.output_tensor))