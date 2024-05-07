import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load the model
model = load_model('/content/drive/MyDrive/Colab Notebooks/h5/trained_model_3.h5')  # Replace with the actual path to your model file

# Load the CIFAR-10 test dataset
(_, _), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
test_images = test_images.astype('float32') / 255.0

# Ensure that the test labels are in the correct format
test_labels = test_labels.squeeze()  # Remove unnecessary dimensions if present

# Convert labels to one-hot encoding
test_labels = to_categorical(test_labels, 10)  # Convert labels to one-hot, assuming 10 classes

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")