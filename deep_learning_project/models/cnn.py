
import tensorflow as tf


'''
create_model function:
 - Returns a compiled CNN model for MNIST digit classification
 - Uses TensorFlow's Keras API to define the model architecture
 - Model consists of convolutional layers, max pooling, flattening, and dense layers
 - Input shape is (28, 28, 1) for grayscale images
 - Output layer has 10 units for the 10 classes (digits 0-9)
 - Uses ReLU activation for hidden layers and softmax for output layer
 - Model is compiled with Adam optimizer and sparse categorical crossentropy loss function
 - Model will be trained on the MNIST dataset loaded from the loader module
 - Model will be saved in HDF5 format after training
 - This function can be used in the training script to create and compile the model
 - The model can be evaluated using the evaluate script after training
 - The model can be loaded from the saved HDF5 file for further use or evaluation
 - This function is essential for building the CNN model architecture for the MNIST dataset
 - The model will be trained for 5 epochs in the training script
 - The model will be evaluated on the test set after training
 - The model will output the test accuracy after evaluation
 - The model can be used for inference on new MNIST images after training and evaluation
 - The model can be modified to include more layers or different architectures if needed
 - The model can be saved and loaded for future use without retraining
 - The model can be used as a baseline for further experiments with different architectures or hyperparameters          
 - Define a simple CNN model for MNIST classification
'''
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
