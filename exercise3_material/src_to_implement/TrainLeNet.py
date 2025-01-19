import os
from Layers import Helpers
from Models.LeNet import build
from NeuralNetwork import NeuralNetwork  # Import the NeuralNetwork class
from Optimization.Loss import CrossEntropyLoss
import matplotlib.pyplot as plt

# Set batch size
batch_size = 50

# Load MNIST dataset
mnist = Helpers.MNISTData(batch_size)
mnist.show_random_training_image()  # Display a random training image

# Create the 'trained' directory if it does not exist
os.makedirs('trained', exist_ok=True)

# Check if a trained model already exists and is not empty
model_path = os.path.join('trained', 'LeNet')
if os.path.isfile(model_path) and os.path.getsize(model_path) > 0:
    # Load the trained model
    net = NeuralNetwork.load(model_path, mnist)  # Call load on the class
else:
    # Build the LeNet model
    net = build()
    net.data_layer = mnist  # Set the data layer
    net.loss_layer = CrossEntropyLoss()  # Set the loss layer

# Train the model
net.train(100)

# Save the trained model
net.save(model_path)  # Call save on the instance

# Plot the loss function
plt.figure('Loss function for training LeNet on the MNIST dataset')
plt.plot(net.loss, '-x')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Test the model
data, labels = net.data_layer.get_test_set()
results = net.test(data)

# Calculate accuracy
accuracy = Helpers.calculate_accuracy(results, labels)
print('\nOn the MNIST dataset, we achieve an accuracy of: {:.2f}%'.format(accuracy * 100))

# from Layers import Helpers
# from Models.LeNet import build
# import NeuralNetwork
# import matplotlib.pyplot as plt
# import os.path

# # Set batch size
# batch_size = 50

# # Load MNIST dataset
# mnist = Helpers.MNISTData(batch_size)
# mnist.show_random_training_image()  # Display a random training image

# # Check if a trained model already exists
# if os.path.isfile(os.path.join('trained', 'LeNet')):
#     # Load the trained model
#     net = NeuralNetwork.load(os.path.join('trained', 'LeNet'), mnist)
# else:
#     # Build the LeNet model
#     net = build()
#     net.data_layer = mnist  # Set the data layer

# # Train the model
# net.train(300)

# # Save the trained model
# NeuralNetwork.save(os.path.join('trained', 'LeNet'), net)

# # Plot the loss function
# plt.figure('Loss function for training LeNet on the MNIST dataset')
# plt.plot(net.loss, '-x')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.show()

# # Test the model
# data, labels = net.data_layer.get_test_set()
# results = net.test(data)

# # Calculate accuracy
# accuracy = Helpers.calculate_accuracy(results, labels)
# print('\nOn the MNIST dataset, we achieve an accuracy of: {:.2f}%'.format(accuracy * 100))