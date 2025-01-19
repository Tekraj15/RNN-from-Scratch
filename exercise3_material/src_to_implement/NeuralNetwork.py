# import numpy as np
# import pickle

# class NeuralNetwork:
#     def __init__(self, optimizer, initializer, initializer_bias):
#         """
#         Neural Network class.
#         :param optimizer: Optimizer object (e.g., SGD, Adam).
#         :param initializer: Weight initializer object (e.g., He, Xavier).
#         :param initializer_bias: Bias initializer object (e.g., Constant).
#         """
#         self.layers = []  # List to store the layers of the network
#         self.loss = []  # List to store the loss values during training
#         self.data_layer = None  # Data layer to provide input and labels
#         self.loss_layer = None  # Loss layer to compute the loss
#         self.optimizer = optimizer  # Optimizer for updating weights
#         self.initializer = initializer  # Weight initializer
#         self.initializer_bias = initializer_bias  # Bias initializer

#     def append_layer(self, layer):
#         """
#         Adds a layer to the network.
#         :param layer: Layer object (e.g., FullyConnected, ReLU, SoftMax).
#         """
#         if layer.trainable:
#             # Initialize weights and biases if the layer is trainable
#             layer.initialize(self.initializer, self.initializer_bias)
#             layer.optimizer = self.optimizer  # Assign the optimizer to the layer
#         self.layers.append(layer)

#     # def forward(self):
#     #     """
#     #     Performs the forward pass through the network.
#     #     :return: Output of the network.
#     #     """
#     #     input_tensor, label_tensor = self.data_layer.next()  # Get input and labels
#     #     output_tensor = input_tensor

#     #     # Pass input through all layers
#     #     for layer in self.layers:
#     #         output_tensor = layer.forward(output_tensor)

#     #     # Compute the loss
#     #     loss = self.loss_layer.forward(output_tensor, label_tensor)
#     #     self.loss.append(loss)  # Store the loss

#     #     return output_tensor

#     def forward(self):
#         """
#         Performs the forward pass through the network.
#         :return: Output of the network.
#         """
#         # Get input and labels from the data layer
#         input_tensor, self.label_tensor = self.data_layer.next()  # Store label_tensor as an attribute
#         output_tensor = input_tensor

#         # Pass input through all layers
#         for layer in self.layers:
#             output_tensor = layer.forward(output_tensor)

#         # Compute the loss
#         loss = self.loss_layer.forward(output_tensor, self.label_tensor)
#         self.loss.append(loss)  # Store the loss

#         return output_tensor

#     # def backward(self):
#     #     """
#     #     Performs the backward pass through the network.
#     #     :return: Gradient of the loss with respect to the input.
#     #     """
        
#     #     #Changed here:
#     #     #error_tensor = self.loss_layer.backward(self.data_layer.label_tensor)

#     #     # Use the label_tensor from the forward pass
#     #     error_tensor = self.loss_layer.backward(self.label_tensor)

#     #     # Propagate the error backward through all layers
#     #     for layer in reversed(self.layers):
#     #         error_tensor = layer.backward(error_tensor)

#     def backward(self):
#         """
#         Performs the backward pass through the network.
#         :return: Gradient of the loss with respect to the input.
#         """
#         # Use the label_tensor stored during the forward pass
#         error_tensor = self.loss_layer.backward(self.label_tensor)

#         # Propagate the error backward through all layers
#         for layer in reversed(self.layers):
#             error_tensor = layer.backward(error_tensor)

#     def train(self, iterations):
#         """
#         Trains the network for a specified number of iterations.
#         :param iterations: Number of training iterations.
#         """
#         for _ in range(iterations):
#             self.forward()  # Forward pass
#             self.backward()  # Backward pass

#     def test(self, input_tensor):
#         """
#         Tests the network on a given input tensor.
#         :param input_tensor: Input tensor for testing.
#         :return: Output of the network.
#         """
#         output_tensor = input_tensor

#         # Pass input through all layers
#         for layer in self.layers:
#             output_tensor = layer.forward(output_tensor)

#         return output_tensor

#     def save(self, filename):
#         """
#         Saves the network to a file using pickle.
#         :param filename: Name of the file to save the network.
#         """
#         with open(filename, 'wb') as f:
#             pickle.dump(self, f)

#     @staticmethod
#     def load(filename, data_layer):
#         """
#         Loads the network from a file using pickle.
#         :param filename: Name of the file to load the network from.
#         :param data_layer: Data layer to provide input and labels.
#         :return: Loaded NeuralNetwork object.
#         """
#         with open(filename, 'rb') as f:
#             net = pickle.load(f)
#         net.data_layer = data_layer  # Set the data layer
#         return net



import pickle

class NeuralNetwork:
    def __init__(self, optimizer, initializer, initializer_bias):
        """
        Neural Network class.
        :param optimizer: Optimizer object (e.g., SGD, Adam).
        :param initializer: Weight initializer object (e.g., He, Xavier).
        :param initializer_bias: Bias initializer object (e.g., Constant).
        """
        self.layers = []  # List to store the layers of the network
        self.loss = []  # List to store the loss values during training
        self.data_layer = None  # Data layer to provide input and labels
        self.loss_layer = None  # Loss layer to compute the loss
        self.optimizer = optimizer  # Optimizer for updating weights
        self.initializer = initializer  # Weight initializer
        self.initializer_bias = initializer_bias  # Bias initializer

    def __getstate__(self):
        """
        Define what to serialize when saving the object.
        Exclude non-picklable objects like generators.
        """
        state = self.__dict__.copy()
        # Exclude non-picklable objects
        if 'data_layer' in state and hasattr(state['data_layer'], '_current_forward_idx_iterator'):
            del state['data_layer']._current_forward_idx_iterator
        return state

    def __setstate__(self, state):
        """
        Define how to restore the object after loading.
        Reinitialize non-picklable objects like generators.
        """
        self.__dict__.update(state)
        # Reinitialize non-picklable objects
        if hasattr(self.data_layer, '_forward_idx_iterator'):
            self.data_layer._current_forward_idx_iterator = self.data_layer._forward_idx_iterator()

    def append_layer(self, layer):
        """
        Adds a layer to the network.
        :param layer: Layer object (e.g., FullyConnected, ReLU, SoftMax).
        """
        if layer.trainable:
            # Initialize weights and biases if the layer is trainable
            layer.initialize(self.initializer, self.initializer_bias)
            layer.optimizer = self.optimizer  # Assign the optimizer to the layer
        self.layers.append(layer)

    def forward(self):
        """
        Performs the forward pass through the network.
        :return: Output of the network.
        """
        input_tensor, self.label_tensor = self.data_layer.next()  # Get input and labels
        output_tensor = input_tensor

        # Pass input through all layers
        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)

        # Compute the loss
        loss = self.loss_layer.forward(output_tensor, self.label_tensor)
        self.loss.append(loss)  # Store the loss

        return output_tensor

    def backward(self):
        """
        Performs the backward pass through the network.
        :return: Gradient of the loss with respect to the input.
        """
        # Use the label_tensor stored during the forward pass
        error_tensor = self.loss_layer.backward(self.label_tensor)

        # Propagate the error backward through all layers
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def train(self, iterations):
        """
        Trains the network for a specified number of iterations.
        :param iterations: Number of training iterations.
        """
        for _ in range(iterations):
            self.forward()  # Forward pass
            self.backward()  # Backward pass

    def test(self, input_tensor):
        """
        Tests the network on a given input tensor.
        :param input_tensor: Input tensor for testing.
        :return: Output of the network.
        """
        output_tensor = input_tensor

        # Pass input through all layers
        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)

        return output_tensor

    def save(self, filename):
        """
        Saves the network to a file using pickle.
        :param filename: Name of the file to save the network.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename, data_layer):
        """
        Loads the network from a file using pickle.
        :param filename: Name of the file to load the network from.
        :param data_layer: Data layer to provide input and labels.
        :return: Loaded NeuralNetwork object.
        """
        with open(filename, 'rb') as f:
            net = pickle.load(f)
        net.data_layer = data_layer  # Set the data layer
        return net