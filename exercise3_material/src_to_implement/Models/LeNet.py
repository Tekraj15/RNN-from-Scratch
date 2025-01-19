from Layers import FullyConnected, ReLU, SoftMax, Flatten, Conv, Pooling
from NeuralNetwork import NeuralNetwork
from Optimization.Optimizers import Adam
from Layers.Initializers import He, Constant

# def build():
#     """
#     Builds the LeNet model.
#     :return: A NeuralNetwork object representing the LeNet model.
#     """
#     # Define the layers
#     layers = [
#         Conv((1, 1), (1, 5, 5), 6),  # Conv layer with 6 filters of size 5x5
#         ReLU(),  # ReLU activation
#         Pooling((2, 2), (2, 2)),  # Max pooling with 2x2 window and stride 2
#         Conv((1, 1), (6, 5, 5), 16),  # Conv layer with 16 filters of size 5x5
#         ReLU(),  # ReLU activation
#         Pooling((2, 2), (2, 2)),  # Max pooling with 2x2 window and stride 2
#         Flatten(),  # Flatten the output for fully connected layers
#         FullyConnected(400, 120),  # Fully connected layer
#         ReLU(),  # ReLU activation
#         FullyConnected(120, 84),  # Fully connected layer
#         ReLU(),  # ReLU activation
#         FullyConnected(84, 10),  # Fully connected layer (output layer)
#         SoftMax()  # SoftMax activation for classification
#     ]

#     # Create the neural network
#     net = NeuralNetwork(
#         optimizer=Adam(learning_rate=5e-4, mu=0.9, rho=0.999),  # Use Adam optimizer
#         initializer=He(),  # Use He initialization for weights
#         initializer_bias=Constant(0.1)  # Initialize biases with 0.1
#     )

#     # Add layers to the network
#     for layer in layers:
#         net.append_layer(layer)

#     return net




def build():
    """
    Builds the LeNet model.
    :return: A NeuralNetwork object representing the LeNet model.
    """
    # Define the layers
    layers = [
        Conv((1, 1), (1, 5, 5), 6),  # Conv layer with 6 filters of size 5x5
        ReLU(),  # ReLU activation
        Pooling((2, 2), (2, 2)),  # Max pooling with 2x2 window and stride 2
        Conv((1, 1), (6, 5, 5), 25),  # Conv layer with 25 filters of size 5x5
        ReLU(),  # ReLU activation
        Pooling((2, 2), (2, 2)),  # Max pooling with 2x2 window and stride 2
        Flatten(),  # Flatten the output for fully connected layers
        FullyConnected(400, 120),  # Fully connected layer
        ReLU(),  # ReLU activation
        FullyConnected(120, 84),  # Fully connected layer
        ReLU(),  # ReLU activation
        FullyConnected(84, 10),  # Fully connected layer (output layer)
        SoftMax()  # SoftMax activation for classification
    ]

    # Create the neural network
    net = NeuralNetwork(
        optimizer=Adam(learning_rate=5e-4, mu=0.9, rho=0.999),  # Use Adam optimizer
        initializer=He(),  # Use He initialization for weights
        initializer_bias=Constant(0.1)  # Initialize biases with 0.1
    )

    # Add layers to the network
    for layer in layers:
        net.append_layer(layer)

    return net